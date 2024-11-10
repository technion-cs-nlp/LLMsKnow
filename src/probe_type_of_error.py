import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.utils import resample

from compute_correctness import  CORRECTNESS_FN
from probe import init_and_train_classifier
from probing_utils import load_model_and_validate_gpu, LIST_OF_DATASETS, compile_probing_indices, \
    get_probing_layer_names, extract_internal_reps_specific_layer_and_token, LIST_OF_MODELS, MODEL_FRIENDLY_NAMES, \
    LIST_OF_PROBING_LOCATIONS, prepare_for_probing, compute_metrics_probing
from resamples_utils import get_error_stats, get_types_of_mistakes


def parse_args_and_init_wandb():
    parser = argparse.ArgumentParser(
        description='Probe for hallucinations and create plots')
    parser.add_argument("--model", choices=LIST_OF_MODELS, required=True)
    parser.add_argument("--probe_at",
                        choices=LIST_OF_PROBING_LOCATIONS, required=True)
    parser.add_argument("--seeds", nargs='+', type=int)
    parser.add_argument("--n_samples", default='all')
    parser.add_argument("--n_resamples", default=30, type=int)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--dataset", choices=LIST_OF_DATASETS, required=True)
    parser.add_argument("--merge_types", action='store_true', default=False)

    args = parser.parse_args()

    wandb.init(
        project="probe_type_of_error",
        config=vars(args)
    )

    return args

def merge_types(TYPES_OF_MISTAKES, TYPES_OF_MISTAKES_TEST):
    NUANCED_TYPES_OF_MISTAKES = {}
    NUANCED_TYPES_OF_MISTAKES_TEST = {}
    TYPES_OF_MISTAKES["wrong_is_largest"] = TYPES_OF_MISTAKES["wrong_is_largest_1"] | TYPES_OF_MISTAKES[
        "wrong_is_largest_2"]
    TYPES_OF_MISTAKES["many_different_answers"] = TYPES_OF_MISTAKES["many_different_answers_1"] | TYPES_OF_MISTAKES[
        "many_different_answers_2"]
    TYPES_OF_MISTAKES["right_is_largest"] = TYPES_OF_MISTAKES["right_is_largest_1"] | TYPES_OF_MISTAKES[
        "right_is_largest_2"]
    NUANCED_TYPES_OF_MISTAKES["wrong_is_largest"] = (
    TYPES_OF_MISTAKES.pop("wrong_is_largest_1"), TYPES_OF_MISTAKES.pop("wrong_is_largest_2"))
    NUANCED_TYPES_OF_MISTAKES["many_different_answers"] = (
    TYPES_OF_MISTAKES.pop("many_different_answers_1"), TYPES_OF_MISTAKES.pop("many_different_answers_2"))
    NUANCED_TYPES_OF_MISTAKES["right_is_largest"] = (
    TYPES_OF_MISTAKES.pop("right_is_largest_1"), TYPES_OF_MISTAKES.pop("right_is_largest_2"))
    TYPES_OF_MISTAKES_TEST["wrong_is_largest"] = TYPES_OF_MISTAKES_TEST["wrong_is_largest_1"] | TYPES_OF_MISTAKES_TEST[
        "wrong_is_largest_2"]
    TYPES_OF_MISTAKES_TEST["many_different_answers"] = TYPES_OF_MISTAKES_TEST["many_different_answers_1"] | \
                                                       TYPES_OF_MISTAKES_TEST[
                                                           "many_different_answers_2"]
    TYPES_OF_MISTAKES_TEST["right_is_largest"] = TYPES_OF_MISTAKES_TEST["right_is_largest_1"] | TYPES_OF_MISTAKES_TEST[
        "right_is_largest_2"]
    NUANCED_TYPES_OF_MISTAKES_TEST["wrong_is_largest"] = (
    TYPES_OF_MISTAKES_TEST.pop("wrong_is_largest_1"), TYPES_OF_MISTAKES_TEST.pop("wrong_is_largest_2"))
    NUANCED_TYPES_OF_MISTAKES_TEST["many_different_answers"] = (
    TYPES_OF_MISTAKES_TEST.pop("many_different_answers_1"), TYPES_OF_MISTAKES_TEST.pop("many_different_answers_2"))
    NUANCED_TYPES_OF_MISTAKES_TEST["right_is_largest"] = (
    TYPES_OF_MISTAKES_TEST.pop("right_is_largest_1"), TYPES_OF_MISTAKES_TEST.pop("right_is_largest_2"))
    return TYPES_OF_MISTAKES, TYPES_OF_MISTAKES_TEST, NUANCED_TYPES_OF_MISTAKES, NUANCED_TYPES_OF_MISTAKES_TEST

def print_stats(TYPES_OF_MISTAKES):
    all_error_types = TYPES_OF_MISTAKES["no_answer_is_largest"]
    for type_of_mistake in TYPES_OF_MISTAKES:
        if type_of_mistake != 'all':
            all_error_types = all_error_types | TYPES_OF_MISTAKES[type_of_mistake]
        print(type_of_mistake, "No. of samples: ", np.sum(TYPES_OF_MISTAKES[type_of_mistake]))
    print("Portion from total answers where the model made at least one error:",
          np.sum(all_error_types) / TYPES_OF_MISTAKES['all'].sum())

def fit_predict(NUANCED_TYPES_OF_MISTAKES, NUANCED_TYPES_OF_MISTAKES_TEST, TYPES_OF_MISTAKES,
                TYPES_OF_MISTAKES_TEST, X_test, X_train, X_valid, args, training_data_indices,
                validation_data_indices, seed):
    results = {}
    for type_of_mistake in TYPES_OF_MISTAKES:
        if type_of_mistake == 'all':
            continue
        print(f"######## {type_of_mistake} #########")
        y_train = TYPES_OF_MISTAKES[type_of_mistake][training_data_indices]
        y_valid = TYPES_OF_MISTAKES[type_of_mistake][validation_data_indices]
        y_test = TYPES_OF_MISTAKES_TEST[type_of_mistake]
        clf = init_and_train_classifier(seed, X_train, y_train)
        metrics = compute_metrics_probing(clf, X_valid, y_valid, pos_label=1)
        metrics_test = compute_metrics_probing(clf, X_test, y_test, pos_label=1)
        for m in metrics:
            results[f'{type_of_mistake}_{m}'] = metrics[m]
        for m in metrics_test:
            results[f'{type_of_mistake}_{m}_test'] = metrics_test[m]
        # for m in metrics:
        #     wandb.summary[f'{type_of_mistake}_{m}'] = metrics[m]
        # for m in metrics_test:
        #     wandb.summary[f'{type_of_mistake}_{m}_test'] = metrics_test[m]

    if args.merge_types:
        for type_of_mistake in NUANCED_TYPES_OF_MISTAKES:
            if type_of_mistake == 'all':
                continue
            print(f"######## BETWEEN TYPES {type_of_mistake} #########")

            X_train_ = X_train[TYPES_OF_MISTAKES[type_of_mistake][training_data_indices]]
            X_valid_ = X_valid[TYPES_OF_MISTAKES[type_of_mistake][validation_data_indices]]
            X_test_ = X_test[TYPES_OF_MISTAKES_TEST[type_of_mistake]]
            y_train_ = NUANCED_TYPES_OF_MISTAKES[type_of_mistake][0][training_data_indices][
                TYPES_OF_MISTAKES[type_of_mistake][training_data_indices]]
            y_valid_ = NUANCED_TYPES_OF_MISTAKES[type_of_mistake][0][validation_data_indices][
                TYPES_OF_MISTAKES[type_of_mistake][validation_data_indices]]
            y_test_ = NUANCED_TYPES_OF_MISTAKES_TEST[type_of_mistake][0][TYPES_OF_MISTAKES_TEST[type_of_mistake]]
            clf = init_and_train_classifier(seed, X_train_, y_train_)
            metrics = compute_metrics_probing(clf, X_valid_, y_valid_, pos_label=1)
            metrics_test = compute_metrics_probing(clf, X_test_, y_test_, pos_label=1)
            for m in metrics:
                results[f'between_types_{type_of_mistake}_{m}'] = metrics[m]
                # wandb.summary[f'between_types_{type_of_mistake}_{m}'] = metrics[m]
            for m in metrics_test:
                results[f'between_types_{type_of_mistake}_{m}_test'] = metrics_test[m]
                # wandb.summary[f'between_types_{type_of_mistake}_{m}_test'] = metrics_test[m]

    return results

def get_internal_reps(args, input_output_ids_greedy, input_output_ids_greedy_test, model, model_output_greedy,
                      model_output_greedy_test, tokenizer, training_data_indices, validation_data_indices):
    data_train, data_valid, input_output_ids_train, input_output_ids_valid, y_train, y_valid, \
        exact_answer_train, exact_answer_valid, validity_of_exact_answer_train, validity_of_exact_answer_valid, \
        questions_train, questions_valid = prepare_for_probing(model_output_greedy, input_output_ids_greedy,
                                                               training_data_indices, validation_data_indices)
    X_train = \
        extract_internal_reps_specific_layer_and_token(model, tokenizer, questions_train,
                                                       input_output_ids_train, args.probe_at, args.model, args.layer,
                                                       args.token,
                                                       exact_answer_train, validity_of_exact_answer_train,
                                                       )
    X_train = np.array(X_train)
    X_valid = \
        extract_internal_reps_specific_layer_and_token(model, tokenizer, questions_valid,
                                                       input_output_ids_valid, args.probe_at, args.model, args.layer,
                                                       args.token,
                                                       exact_answer_valid, validity_of_exact_answer_valid)
    X_valid = np.array(X_valid)
    questions_test = model_output_greedy_test['question']
    X_test = extract_internal_reps_specific_layer_and_token(model, tokenizer, questions_test,
                                                            input_output_ids_greedy_test, args.probe_at, args.model,
                                                            args.layer, args.token,
                                                            model_output_greedy_test['exact_answer'],
                                                            model_output_greedy_test['valid_exact_answer'])
    X_test = np.array(X_test)

    return X_train, X_valid, X_test


def main():

    args = parse_args_and_init_wandb()

    textual_answers = torch.load(f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.n_resamples}_textual_answers.pt")
    exact_answers = torch.load(f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.n_resamples}_exact_answers.pt")
    model_output_greedy = pd.read_csv(f'../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv')
    input_output_ids_greedy = torch.load(
            f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{args.dataset}.pt")

    textual_answers_test = torch.load(f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_test_{args.n_resamples}_textual_answers.pt")
    exact_answers_test = torch.load(f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_test_{args.n_resamples}_exact_answers.pt")
    model_output_greedy_test = pd.read_csv(f'../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}_test.csv')
    input_output_ids_greedy_test = torch.load(
            f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{args.dataset}_test.pt")

    error_stats = get_error_stats(textual_answers, exact_answers, model_output_greedy, CORRECTNESS_FN['triviaqa'])
    error_stats = pd.DataFrame.from_dict(error_stats)
    error_stats.to_csv(f"{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_clusters.csv")
    TYPES_OF_MISTAKES = get_types_of_mistakes(error_stats, args.n_resamples)

    error_stats_test = get_error_stats(textual_answers_test, exact_answers_test, model_output_greedy_test, CORRECTNESS_FN['triviaqa'])
    error_stats_test = pd.DataFrame.from_dict(error_stats_test)
    error_stats_test.to_csv(f"{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_clusters_test.csv")
    TYPES_OF_MISTAKES_TEST = get_types_of_mistakes(error_stats_test, args.n_resamples)

    NUANCED_TYPES_OF_MISTAKES, NUANCED_TYPES_OF_MISTAKES_TEST = None, None
    if args.merge_types:
        TYPES_OF_MISTAKES, TYPES_OF_MISTAKES_TEST, NUANCED_TYPES_OF_MISTAKES, NUANCED_TYPES_OF_MISTAKES_TEST = merge_types(TYPES_OF_MISTAKES, TYPES_OF_MISTAKES_TEST)

    print_stats(TYPES_OF_MISTAKES)

    model, tokenizer = load_model_and_validate_gpu(args.model)

    results_per_seed = defaultdict(list)
    for seed in args.seeds:
        training_data_indices, validation_data_indices = compile_probing_indices(model_output_greedy, args.n_samples, seed,
                                                                                 n_validation_samples=1000 if args.n_samples == 'all' else min(1000, int(args.n_samples)))
        model_output_greedy_test_, input_output_ids_greedy_test_ = resample(model_output_greedy_test, input_output_ids_greedy_test, random_state=seed)

        X_train, X_valid, X_test = get_internal_reps(args, input_output_ids_greedy, input_output_ids_greedy_test_, model,
                                                     model_output_greedy, model_output_greedy_test_, tokenizer,
                                                     training_data_indices, validation_data_indices)

        results = fit_predict(NUANCED_TYPES_OF_MISTAKES, NUANCED_TYPES_OF_MISTAKES_TEST, TYPES_OF_MISTAKES,
                    TYPES_OF_MISTAKES_TEST, X_test, X_train, X_valid, args, training_data_indices,
                    validation_data_indices, seed)

        for res in results:
            results_per_seed[res].append(results[res])

    for res in results_per_seed:
        wandb.summary[res] = np.mean(results_per_seed[res])
        wandb.summary[f'{res}_std'] = np.std(results_per_seed[res])

if __name__ == "__main__":
    main()