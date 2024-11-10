import argparse
import os
import pickle
from collections import defaultdict
from os.path import exists

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from transformers import set_seed

from probing_utils import extract_internal_reps_specific_layer_and_token, compile_probing_indices, \
    load_model_and_validate_gpu, get_probing_layer_names, LIST_OF_DATASETS, LIST_OF_MODELS, \
    MODEL_FRIENDLY_NAMES, LIST_OF_PROBING_LOCATIONS, compute_metrics_probing, prepare_for_probing


def parse_args_and_init_wandb():
    parser = argparse.ArgumentParser(
        description='Probe for hallucinations and create plots')
    parser.add_argument("--model", choices=LIST_OF_MODELS)
    parser.add_argument("--probe_at",
                        choices=LIST_OF_PROBING_LOCATIONS)
    # important args for the results of the papaer
    parser.add_argument("--seeds", nargs='+', type=int)
    parser.add_argument("--n_samples", help="size of validation data", default='all')
    parser.add_argument("--layer", type=int)
    parser.add_argument("--token", type=str)
    parser.add_argument("--save_clf", action='store_true', default=False, help="Whether to save the clf. If true, will look for a classifier before training and load it if exists.")
    parser.add_argument("--dataset", choices=LIST_OF_DATASETS, required=True)
    parser.add_argument("--test_dataset", choices=LIST_OF_DATASETS, required=False, default=None)

    args = parser.parse_args()

    if args.test_dataset is None:
        wandb.init(
            project="probe_hallucinations_specific",
            config=vars(args)
        )
    else:
        wandb.init(
            project="probe_hallucinations_generalization",
            config=vars(args)
        )

    return args

def probe(model, tokenizer, data, input_output_ids, token, layer, probe_at, seeds,
              model_name, dataset_name, n_samples,
          data_test=None, input_output_ids_test=None, clf=None):

    train_clf = clf is None

    # Data preprocessing
    data_train_valid, _, input_output_ids_train_valid, _, y_train_valid, _, \
        exact_answer_train_valid, _, validity_of_exact_answer_train_valid, _, \
        questions_train_valid, _ = prepare_for_probing(
        data, input_output_ids, np.arange(len(data)) if n_samples == 'all' else np.arange(min(len(data), int(n_samples))), [])
    X_train_valid = None
    if train_clf:
        X_train_valid = \
            extract_internal_reps_specific_layer_and_token(model, tokenizer, questions_train_valid,
                                                           input_output_ids_train_valid, probe_at, model_name,
                                                           layer, token, exact_answer_train_valid,
                                                           validity_of_exact_answer_train_valid,
                                                           )
        X_train_valid = np.array(X_train_valid)

    X_test = None
    y_test = None
    if data_test is not None:
        test_data_indices = data_test.index
        if 'exact_answer' in data:
            test_data_indices = test_data_indices[
                (data_test.iloc[test_data_indices]['valid_exact_answer'] == 1) & (
                            data_test.iloc[test_data_indices]['exact_answer'] != 'NO ANSWER') & (
                            data_test.iloc[test_data_indices]['exact_answer'].map(lambda x: type(x)) == str)]

        _, _, input_output_ids_test, _, y_test, _, \
            exact_answer_test, _, validity_of_exact_answer_test, _, \
            questions_test, _ = prepare_for_probing(
            data_test, input_output_ids_test, test_data_indices, [])
        X_test = extract_internal_reps_specific_layer_and_token(model, tokenizer, questions_test,
                                                                input_output_ids_test, probe_at, model_name, layer,
                                                                token,
                                                                exact_answer_test, validity_of_exact_answer_test)

    valid_metrics_per_seed = defaultdict(list)
    test_metrics_per_seed = defaultdict(list)

    for seed in seeds:
        print(f"##### {seed} #####")
        set_seed(seed)
        n_validation_samples = 1000 if n_samples == 'all' else min(1000, int(n_samples))
        training_data_indices, validation_data_indices = compile_probing_indices(data_train_valid, n_samples,
                                                                                 seed,
                                                                                 n_validation_samples=n_validation_samples)
        if train_clf:
            clf = init_and_train_classifier(seed, X_train_valid, y_train_valid)
            clf_only_train = init_and_train_classifier(seed, X_train_valid[training_data_indices],
                                            y_train_valid[training_data_indices])
            valid_metrics_for_seed = compute_metrics_probing(clf_only_train, X_train_valid[validation_data_indices],
                                                             y_train_valid[validation_data_indices])
            for k in valid_metrics_for_seed:
                valid_metrics_per_seed[k].append(valid_metrics_for_seed[k])

        if data_test is not None:
            X_test_, y_test_ = resample(X_test, y_test, random_state=seed)
            test_metrics_for_seed = compute_metrics_probing(clf, X_test_, y_test_)
            for k in test_metrics_for_seed:
                test_metrics_per_seed[k].append(test_metrics_for_seed[k])

    # compute mean, std per metric
    valid_metrics_aggregated = aggregate_metrics_across_seeds(valid_metrics_per_seed)
    test_metrics_aggregated = aggregate_metrics_across_seeds(test_metrics_per_seed)

    return valid_metrics_aggregated, test_metrics_aggregated, clf


def aggregate_metrics_across_seeds(metrics_per_seed):
    metrics_aggregated = {}
    for k in metrics_per_seed:
        metrics_aggregated[f"{k}"] = np.mean(metrics_per_seed[k])
        metrics_aggregated[f"{k}_std"] = np.std(metrics_per_seed[k])
    return metrics_aggregated


def init_and_train_classifier(seed, X_train, y_train):
    clf = LogisticRegression(random_state=seed).fit(X_train, y_train)
    return clf


def get_saved_clf_if_exists(args):
    if not exists("../checkpoints"):
        os.makedirs("../checkpoints")
    save_path = f"../checkpoints/clf_{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_layer-{args.layer}_token-{args.token}.pkl"
    print("Loading classifier from ", save_path)

    if exists(save_path):
        save_clf = False
        with open(save_path, 'rb') as f:
            clf = pickle.load(f)
    else:
        save_clf = True
        clf = None
        print("Classifier not found, training new one")
    return clf, save_clf, save_path

def main():
    args = parse_args_and_init_wandb()

    model, tokenizer = load_model_and_validate_gpu(args.model)

    data_test = None
    input_output_ids_test = None
    model_output_file = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    data = pd.read_csv(model_output_file).reset_index()
    input_output_ids = torch.load(
        f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{args.dataset}.pt")

    if args.test_dataset is not None:
        test_dataset = args.test_dataset
    else:
        test_dataset = args.dataset
    model_output_file_test = f"../output/mistral-7b-instruct-answers-{test_dataset}_test.csv"
    load_test = False
    if os.path.isfile(model_output_file_test):
        data_test = pd.read_csv(model_output_file_test)
        input_output_ids_test = torch.load(
            f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{test_dataset}_test.pt")
        load_test = True

    if args.save_clf:
        clf, save_clf, save_path = get_saved_clf_if_exists(args)
    else:
        save_clf = False
        clf = None

    res = probe(model, tokenizer, data, input_output_ids, args.token,
                                                   args.layer, args.probe_at, args.seeds, args.model, args.dataset,
                                                    args.n_samples, data_test, input_output_ids_test, clf)

    if load_test:
        metrics_test = res[1]
        for m in metrics_test:
            wandb.summary[f"{m}_test"] = metrics_test[m]

    metrics_valid, _, clf = res
    for m in metrics_valid:
        wandb.summary[m] = metrics_valid[m]

    if save_clf:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
        print("Saved classifier to ", save_path)


if __name__ == "__main__":
    main()