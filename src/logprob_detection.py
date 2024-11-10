import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample
from transformers import AutoTokenizer

from probing_utils import LIST_OF_DATASETS, get_indices_of_exact_answer, compile_probing_indices, \
    LIST_OF_MODELS, MODEL_FRIENDLY_NAMES, compute_metrics_probing
from tqdm import tqdm
import os


def parse_args_and_init_wandb():
    parser = argparse.ArgumentParser(
        description='A baseline for comparing probing. '
                    'Detection is done by looking at the predicted logits of probabilities across the vocabulary.')
    parser.add_argument("--model", choices=LIST_OF_MODELS)
    parser.add_argument("--dataset", choices=LIST_OF_DATASETS, required=True)
    parser.add_argument("--seeds", type=int, nargs='+')
    parser.add_argument("--use_exact_answer", action='store_true', default=False)
    args = parser.parse_args()

    wandb.init(
        project="logprob",
        config=vars(args)
    )

    return args

def load_logits(model_name, dataset, load_test=False):
    print("Loading logits")
    logits_test = None
    train_logits_path = f"../output/{MODEL_FRIENDLY_NAMES[model_name]}-scores-{dataset}.pt"
    if not os.path.exists(train_logits_path):
        print(f"File {train_logits_path} does not exist. Please run the training script first.")
        return
    try:
        logits = torch.load(train_logits_path)
    except Exception as e:
        print(f"Path exists but probably corrupted: {train_logits_path}")
        print(f"Error loading logits: {e}")
        return
    if load_test:
        logits_test = torch.load(f"../output/{MODEL_FRIENDLY_NAMES[model_name]}-scores-{dataset}_test.pt")
    print("Loaded logits")

    return logits, logits_test


def load_input_output_ids(model_name, dataset, load_test=False):
    print("Loading input_output_ids")
    input_output_ids_test = None
    input_output_ids_path = f"../output/{MODEL_FRIENDLY_NAMES[model_name]}-input_output_ids-{dataset}.pt"
    if not os.path.exists(input_output_ids_path):
        print(f"File {input_output_ids_path} does not exist. Please run the training script first.")
        return
    input_output_ids = torch.load(
        f"../output/{MODEL_FRIENDLY_NAMES[model_name]}-input_output_ids-{dataset}.pt")
    if load_test:
        input_output_ids_test = torch.load(
            f"../output/{MODEL_FRIENDLY_NAMES[model_name]}-input_output_ids-{dataset}_test.pt")
    print("Loaded input_output_ids")
    return input_output_ids, input_output_ids_test


def main():
    print("Starting detection by logprob")
    args = parse_args_and_init_wandb()
    model_output_file = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    data = pd.read_csv(model_output_file)
    model_output_file_test = f"../output/mistral-7b-instruct-answers-{args.dataset}_test.csv"
    load_test = False
    if os.path.isfile(model_output_file_test):
        data_test = pd.read_csv(model_output_file_test)
        load_test = True

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    logits, logits_test = load_logits(args.model, args.dataset, load_test=load_test)
    input_output_ids, input_output_ids_test = load_input_output_ids(args.model, args.dataset, load_test=load_test)

    all_results = defaultdict(list)
    all_results_test = defaultdict(list)
    for seed in args.seeds:
        print(f"Seed: {seed}")
        _, validation_data_indices = compile_probing_indices(data, 'all', seed,
                                                                                 n_validation_samples=1000)
        data_valid = data.iloc[validation_data_indices]
        results = logprob_detection(tokenizer, data_valid, input_output_ids, logits, args)
        for strategy in results:
            all_results[strategy].append(results[strategy])

        if load_test:
            test_data_indices = data_test.index
            if "exact_answer" in data_test:
                test_data_indices = test_data_indices[
                    (data_test['valid_exact_answer'] == 1) & (
                                data_test['exact_answer'] != 'NO ANSWER') & (data_test['exact_answer'].map(lambda x : type(x)) == str)]
            data_test_ = data_test.iloc[test_data_indices]
            data_test_ = resample(data_test_, random_state=seed)
            results = logprob_detection(tokenizer, data_test_, input_output_ids_test, logits_test, args)
            for strategy in results:
                all_results_test[strategy].append(results[strategy])

    # log results
    log_results(all_results)

    if data_test is not None:
        log_results(all_results_test, test=True)

def log_results(all_results, test=False):
    if test:
        _test = "_test"
    else:
        _test = ""
    for strategy in all_results:
        if len(all_results[strategy]) == 0:
            continue
        for metric in all_results[strategy][0]:
            if all_results[strategy][0][metric] is not None:
                wandb.summary[f"{strategy}.{metric}.mean{_test}"] = np.mean([x[metric] for x in all_results[strategy]])
                wandb.summary[f"{strategy}.{metric}.std{_test}"] = np.std([x[metric] for x in all_results[strategy]])


def logprob_detection(tokenizer, data, input_output_ids, logits, args):
    results = defaultdict(list)
    y = data.automatic_correctness.to_numpy()

    scores_logits_after_last_token, scores_logits_last_token, scores_logits_max, scores_logits_mean, \
        scores_logits_min, scores_probas_after_last_token, scores_probas_last_token, scores_probas_max, \
        scores_probas_mean, scores_probas_min = get_logprob_scores(
        args, data, input_output_ids, logits, tokenizer)
    results["logits_min"] = compute_metrics_probing(None, None, y, pos_label=1, predicted_probas=scores_logits_min)
    results["logits_max"] = compute_metrics_probing(None, None, y, pos_label=1, predicted_probas=scores_logits_max)
    results["logits_mean"] = compute_metrics_probing(None, None, y, pos_label=1, predicted_probas=scores_logits_mean)
    results["logits_last_token"] = compute_metrics_probing(None, None, y, pos_label=1, predicted_probas=scores_logits_last_token)
    if args.use_exact_answer:
        results["logits_after_last_token"] = compute_metrics_probing(None, None, y, pos_label=1, predicted_probas=scores_logits_after_last_token)
    results["probas_min"] = compute_metrics_probing(None, None, y, pos_label=1, predicted_probas=scores_probas_min)
    results["probas_max"] = compute_metrics_probing(None, None, y, pos_label=1, predicted_probas=scores_probas_max)
    results["probas_mean"] = compute_metrics_probing(None, None, y, pos_label=1, predicted_probas=scores_probas_mean)
    results["probas_last_token"] = compute_metrics_probing(None, None, y, pos_label=1,  predicted_probas=scores_probas_last_token)
    if args.use_exact_answer:
        results["probas_after_last_token"] = compute_metrics_probing(None, None, y, predicted_probas=scores_probas_after_last_token)

    return results


def get_logprob_scores(args, data, input_output_ids, logits, tokenizer):
    all_logit_scores_min = []
    all_logit_scores_max = []
    all_logit_scores_mean = []
    all_logit_scores_last_token = []
    all_logit_scores_after_last_token = []
    all_probas_scores_min = []
    all_probas_scores_max = []
    all_probas_scores_mean = []
    all_probas_scores_last_token = []
    all_probas_scores_after_last_token = []
    for idx, row in tqdm(data.iterrows()):
        tokenized_full_answer = input_output_ids[idx]
        logits_for_ex = logits['all_scores'][idx]
        probas_for_ex = logits_for_ex.softmax(dim=1)
        output_ids = logits['all_output_ids'][idx]
        logits_for_ex_generated_tokens = logits_for_ex.gather(1, output_ids.unsqueeze(1)).view(
            logits_for_ex.shape[0])
        probas_for_ex_generated_tokens = probas_for_ex.gather(1, output_ids.unsqueeze(1)).view(
            probas_for_ex.shape[0])

        if args.use_exact_answer:
            exact_answer = row['exact_answer']

            exact_indices_for_probing = get_indices_of_exact_answer(tokenizer, tokenized_full_answer,
                                                                    str(exact_answer), output_ids=output_ids, model_name=args.model)

            exact_indices_for_logits = torch.tensor(exact_indices_for_probing) - (
                    tokenized_full_answer.shape[0] - output_ids.shape[0])
            analyzed_logits = logits_for_ex_generated_tokens[exact_indices_for_logits]
            analyzed_probas = probas_for_ex_generated_tokens[exact_indices_for_logits]
            logit_after_last_question_token = logits_for_ex_generated_tokens[
                min(exact_indices_for_logits[-1] + 1, len(logits_for_ex_generated_tokens) - 1)]
            probas_after_last_question_token = probas_for_ex_generated_tokens[
                min(exact_indices_for_logits[-1] + 1, len(probas_for_ex_generated_tokens) - 1)]

            all_logit_scores_after_last_token.append(logit_after_last_question_token.item())
            all_probas_scores_after_last_token.append(probas_after_last_question_token.item())
        else:
            analyzed_logits = logits_for_ex_generated_tokens
            analyzed_probas = probas_for_ex_generated_tokens

        all_logit_scores_min.append(analyzed_logits.min().item())
        all_logit_scores_max.append(analyzed_logits.max().item())
        all_logit_scores_mean.append(analyzed_logits.mean().item())
        all_logit_scores_last_token.append(analyzed_logits[-1].item())

        all_probas_scores_min.append(analyzed_probas.min().item())
        all_probas_scores_max.append(analyzed_probas.max().item())
        all_probas_scores_mean.append(analyzed_probas.mean().item())
        all_probas_scores_last_token.append(analyzed_probas[-1].item())

    all_logit_scores_min = np.array(all_logit_scores_min)
    all_logit_scores_max = np.array(all_logit_scores_max)
    all_logit_scores_mean = np.array(all_logit_scores_mean)
    all_logit_scores_last_token = np.array(all_logit_scores_last_token)
    all_logit_scores_after_last_token = np.array(all_logit_scores_after_last_token)
    all_probas_scores_min = np.array(all_probas_scores_min)
    all_probas_scores_max = np.array(all_probas_scores_max)
    all_probas_scores_mean = np.array(all_probas_scores_mean)
    all_probas_scores_last_token = np.array(all_probas_scores_last_token)
    all_probas_scores_after_last_token = np.array(all_probas_scores_after_last_token)
    return all_logit_scores_after_last_token, all_logit_scores_last_token, all_logit_scores_max, all_logit_scores_mean, all_logit_scores_min, all_probas_scores_after_last_token, all_probas_scores_last_token, all_probas_scores_max, all_probas_scores_mean, all_probas_scores_min


if __name__ == "__main__":
    main()