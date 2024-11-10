import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.utils import resample

from compute_correctness import CORRECTNESS_FN
from generate_model_answers import MODEL_FRIENDLY_NAMES
from probing_utils import load_model_and_validate_gpu, extract_internal_reps_specific_layer_and_token, \
    get_probing_layer_names, LIST_OF_DATASETS, compute_metrics_probing, \
    LIST_OF_MODELS, LIST_OF_PROBING_LOCATIONS, LIST_OF_TEST_DATASETS
from resamples_utils import get_error_stats, get_types_of_mistakes


def parse_args_and_init_wandb():
    parser = argparse.ArgumentParser(
        description='Measure how well the model does with probe choice')
    parser.add_argument("--model", choices=LIST_OF_MODELS)
    parser.add_argument("--probe_at",
                        choices=LIST_OF_PROBING_LOCATIONS)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--token", type=str)
    parser.add_argument("--dataset",
                        choices=LIST_OF_TEST_DATASETS)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument("--n_resamples", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=None, help="In case you want to use a subset of the data for the analysis")
    args = parser.parse_args()

    wandb.init(
        project="probe_choose_answer",
        config=vars(args)
    )
    return args


def compute_correctness_majority(n_questions, error_stats):
    correctness_with_majority = []
    for ex_idx in range(n_questions):
        if error_stats[ex_idx]['correct_answer_size'] > error_stats[ex_idx]['largest_incorrect_answer_size']:
            correctness_with_majority.append(1)
        else:
            correctness_with_majority.append(0)
    return np.array(correctness_with_majority)


def compute_correctness_random(n_questions, all_exact_answers, correctness_per_resample):
    correctness_with_random_choosing = []
    for ex_idx in range(n_questions):

        valid_answer_resamples = np.where(np.array(all_exact_answers['valid_exact_answer'][ex_idx]) == 1)[0]
        if len(valid_answer_resamples) == 0:
            correctness = 0
        else:
            randomized_resample_idx = np.random.choice(valid_answer_resamples)
            correctness = correctness_per_resample[randomized_resample_idx][ex_idx]
        correctness_with_random_choosing.append(correctness)

    correctness_with_random_choosing = np.array(correctness_with_random_choosing)
    correctness_with_random_choosing_mean = correctness_with_random_choosing.mean()

    print("correctness_with_random_choosing:", "%.2f" % correctness_with_random_choosing_mean)
    return correctness_with_random_choosing


def get_probe_pred_per_resample(model, tokenizer, clf, layer, token, model_output_greedy, all_textual_answers,
                                all_input_output_ids, all_exact_answers_per_resample,
                                all_exact_answers_validity_per_resample,
                                probe_at, model_name, compute_correctness_fn, limit_samples=None):
    probe_preds_per_resample = []
    correctness_per_resample = []
    metrics_per_resample = []

    # iterate over different resamples
    for textual_answers, input_output_ids, exact_answer, validity_of_exact_answer in zip(all_textual_answers,
                                                                                         all_input_output_ids,
                                                                                         all_exact_answers_per_resample,
                                                                                         all_exact_answers_validity_per_resample):
        questions = model_output_greedy.question
        correct_answers = model_output_greedy.correct_answer

        X = \
            extract_internal_reps_specific_layer_and_token(model, tokenizer, questions[:limit_samples],
                                                           input_output_ids[:limit_samples], probe_at, model_name,
                                                           layer, token, exact_answer[:limit_samples],
                                                           validity_of_exact_answer[:limit_samples],
                                                           use_dict_for_tokens=False
                                                           )

        if 'incorrect_answer' in model_output_greedy:
            correctness = compute_correctness_fn(textual_answers[:limit_samples], [correct_answers,
                                                                                   model_output_greedy.incorrect_answer])[
                'correctness']
        else:
            correctness = compute_correctness_fn(textual_answers[:limit_samples], correct_answers.to_numpy())[
                'correctness']
        correctness = np.array(correctness)
        probe_preds_per_resample.append(clf.predict_proba(X))
        correctness_per_resample.append(correctness)
        metrics_per_resample.append(compute_metrics_probing(clf, X, correctness))
    return probe_preds_per_resample, correctness_per_resample, metrics_per_resample

def compute_correctness_with_probing(n_questions, n_resamples, all_exact_answers, probe_preds_per_resample,
                                     correctness_per_resample, all_exact_answers_per_resample, all_textual_answers):
    correctness_with_probing = []
    chosen_answer = []
    chosen_answer_full = []
    correctness_scores = []

    for i, ex_idx in enumerate(range(n_questions)):
        valid_answer_resamples = np.array(all_exact_answers['valid_exact_answer'][
                                              ex_idx]) == 1  # only keeping valid answers because a non valid answer means wrong answer
        probe_preds = [probe_preds_per_resample[resample_idx][i][1] if valid_answer_resamples[resample_idx] == 1 else 0 for
                       resample_idx in range(n_resamples)]
        max_idx = np.argmax(probe_preds)

        correctness = correctness_per_resample[max_idx][i]
        correctness_with_probing.append(correctness)
        chosen_answer.append(all_exact_answers_per_resample[max_idx][ex_idx])
        chosen_answer_full.append(all_textual_answers[max_idx][ex_idx])
        correctness_scores.append(np.max(probe_preds))

    print("correctness_with_probing:", np.mean(correctness_with_probing))
    correctness_with_probing = np.array(correctness_with_probing)
    chosen_answer = np.array(chosen_answer)
    chosen_answer_full = np.array(chosen_answer_full)
    correctness_scores = np.array(correctness_scores)
    return correctness_with_probing, chosen_answer, chosen_answer_full, correctness_scores


def main():
    args = parse_args_and_init_wandb()

    model, tokenizer = load_model_and_validate_gpu(args.model)

    model_output_file = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    model_output_greedy = pd.read_csv(model_output_file).reset_index(drop=True)
    model_output_greedy = model_output_greedy[model_output_greedy['valid_exact_answer'] == 1]

    #temp
    print("Greedy correctness", model_output_greedy.automatic_correctness.mean())

    textual_answers = torch.load(
        f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.n_resamples}_textual_answers.pt")
    textual_answers = [[textual_answers[j][i] for i in model_output_greedy.index] for j in range(args.n_resamples)]
    n_questions = len(model_output_greedy) if args.n_samples is None else args.n_samples

    exact_answers_and_validity = torch.load(
        f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.n_resamples}_exact_answers.pt")
    exact_answers_and_validity['exact_answer'] = [[exact_answers_and_validity['exact_answer'][i][j] for j in range(0, args.n_resamples)] for i in model_output_greedy.index]
    exact_answers_and_validity['valid_exact_answer'] = [[exact_answers_and_validity['valid_exact_answer'][i][j] for j in range(0, args.n_resamples)] for i in model_output_greedy.index]

    exact_answers = exact_answers_and_validity['exact_answer']
    valid_exact_answers = exact_answers_and_validity['valid_exact_answer']
    exact_answers_per_resample = [[exact_answers[i][j] for i in range(0, n_questions)] for j in
                                      range(0, args.n_resamples)]
    exact_answers_validity_per_resample = [[valid_exact_answers[i][j] for i in range(0, n_questions)] for j in
                                               range(0, args.n_resamples)]

    input_output_ids = torch.load(
        f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_{args.n_resamples}_input_output_ids.pt")
    input_output_ids = [[input_output_ids[j][i] for i in model_output_greedy.index] for j in range(args.n_resamples)]

    probe_checkpoint = f'../checkpoints/clf_{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset.replace("_test", "")}_layer-{args.layer}_token-{args.token}.pkl'
    with open(probe_checkpoint, 'rb') as f:
        clf = pickle.load(f)

    error_stats = get_error_stats(textual_answers, exact_answers_and_validity, model_output_greedy,
                                  CORRECTNESS_FN[args.dataset.replace("_test", "")])
    types_of_mistakes = get_types_of_mistakes(pd.DataFrame.from_dict(error_stats), args.n_resamples)

    for type_ in types_of_mistakes:
        print(type_, np.mean(types_of_mistakes[type_]), np.sum(types_of_mistakes[type_]))

    probe_preds_per_resample, correctness_per_resample, metrics_per_resample = get_probe_pred_per_resample(model,
                                                                                                           tokenizer,
                                                                                                           clf,
                                                                                                           args.layer,
                                                                                                           args.token,
                                                                                                           model_output_greedy,
                                                                                                           textual_answers,
                                                                                                           input_output_ids,
                                                                                                           exact_answers_per_resample,
                                                                                                           exact_answers_validity_per_resample,
                                                                                                           args.probe_at,
                                                                                                           args.model,
                                                                                                           CORRECTNESS_FN[
                                                                                                               args.dataset.replace(
                                                                                                                   "_test",
                                                                                                                   "")],
                                                                                                           limit_samples=n_questions,
                                                                                                           )
    correctness_with_probing, chosen_answer, chosen_answer_full, correctness_scores = compute_correctness_with_probing(
        n_questions, args.n_resamples, exact_answers_and_validity, probe_preds_per_resample, correctness_per_resample,
        exact_answers_per_resample, textual_answers,
    )

    correctness_with_random_choosing = compute_correctness_random(n_questions, exact_answers_and_validity,
                                                                  correctness_per_resample)

    correctness_with_majority = compute_correctness_majority(n_questions, error_stats)

    for type_ in types_of_mistakes:
        correctness_with_random_choosing_for_type = correctness_with_random_choosing[types_of_mistakes[type_]]
        correctness_with_probing_for_type = correctness_with_probing[types_of_mistakes[type_]]
        correctness_with_greedy_for_type = model_output_greedy.automatic_correctness[types_of_mistakes[type_]]
        correctness_with_majority_for_type = correctness_with_majority[types_of_mistakes[type_]]

        all_correctness_with_random_choosing = []
        all_correctness_with_probing = []
        all_correctness_with_greedy = []
        all_correctness_with_majority = []
        for seed in args.seeds:
            correctness_with_random_choosing_for_type_ = resample(correctness_with_random_choosing_for_type,
                                                                  random_state=seed)
            all_correctness_with_random_choosing.append(correctness_with_random_choosing_for_type_.mean())

            correctness_with_probing_for_type_ = resample(correctness_with_probing_for_type, random_state=seed)
            all_correctness_with_probing.append(correctness_with_probing_for_type_.mean())

            correctness_with_greedy_for_type_ = resample(correctness_with_greedy_for_type, random_state=seed)
            all_correctness_with_greedy.append(correctness_with_greedy_for_type_.mean())

            correctness_with_majority_for_type_ = resample(correctness_with_majority_for_type, random_state=seed)
            all_correctness_with_majority.append(correctness_with_majority_for_type_.mean())

        wandb.summary[f'random_correctness_{type_}'] = np.mean(all_correctness_with_random_choosing)
        wandb.summary[f'random_correctness_std_{type_}'] = np.std(all_correctness_with_random_choosing)

        wandb.summary[f'probing_correctness_{type_}'] = np.mean(all_correctness_with_probing)
        wandb.summary[f'probing_correctness_std_{type_}'] = np.std(all_correctness_with_probing)

        wandb.summary[f'greedy_correctness_{type_}'] = np.mean(all_correctness_with_greedy)
        wandb.summary[f'greedy_correctness_std_{type_}'] = np.std(all_correctness_with_greedy)

        wandb.summary[f'majority_correctness_{type_}'] = np.mean(all_correctness_with_majority)
        wandb.summary[f'majority_correctness_std_{type_}'] = np.std(all_correctness_with_majority)

        wandb.summary[f'percentage_{type_}'] = np.mean(types_of_mistakes[type_])
        wandb.summary[f'count_{type_}'] = np.sum(types_of_mistakes[type_])

    for idx in range(len(error_stats)):
        error_stats[idx]["chosen_answer_probe"] = chosen_answer[idx]
        error_stats[idx]["probe_correctness"] = correctness_with_probing[idx]
        error_stats[idx]["random_choosing_correctness"] = correctness_with_random_choosing[idx]
        error_stats[idx]["majority_correctness"] = correctness_with_majority[idx]
        error_stats[idx]["chosen_answer_full_probe"] = chosen_answer_full[idx]
    pd.DataFrame.from_dict(error_stats).to_csv(
        f"../output/resampling/{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}-{args.n_resamples}_clusters.csv")


if __name__ == "__main__":
    main()
