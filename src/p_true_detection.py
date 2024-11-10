import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.utils import resample

from detection_by_logprob import metric_fn
from probing_utils import LIST_OF_MODELS, LIST_OF_DATASETS, load_model_and_validate_gpu, compute_metrics_probing
from generate_model_answers import generate_model_answers, MODEL_FRIENDLY_NAMES


def parse_args_and_init_wandb():
    parser = argparse.ArgumentParser(
        description='Probe for hallucinations and create plots')
    parser.add_argument("--model", choices=LIST_OF_MODELS)
    parser.add_argument("--seeds", nargs='+', type=int)
    parser.add_argument("--n_samples", default=None)
    parser.add_argument("--dataset", choices=LIST_OF_DATASETS)
    parser.add_argument("--use_exact_answer", default=False, action='store_true')

    args = parser.parse_args()

    wandb.init(
        project="p_true_detection",
        config=vars(args)
    )

    return args

def get_p_true(data, model, tokenizer, model_name, use_exact_answer=False, reversed=False):

    if use_exact_answer:
        answer_col = 'exact_answer'
    else:
        answer_col = 'model_answer'

    question_col = 'raw_question' if 'raw_question' in data.columns else 'question'

    prompts = []
    labels = []
    for question, answer, validity_exact, y in zip(data[question_col], data[answer_col], data['valid_exact_answer'], data['automatic_correctness']):
        if (validity_exact == 0) or (validity_exact == 'NO ANSWER'):
            continue

        if reversed:
            prompt = \
                f"""Question: {question}
        Possible answer: {answer}
        Is the possible answer:
        (A) False
        (B) True
        The possible answer is:"""
        else:
            prompt = \
                f"""Question: {question}
            Possible answer: {answer}
            Is the possible answer:
            (A) True
            (B) False
            The possible answer is:"""

        prompts.append(prompt)
        labels.append(y)

    token_a = tokenizer.encode('A', add_special_tokens=False)[0]
    token_b = tokenizer.encode('B', add_special_tokens=False)[0]
    all_textual_answers, _, all_scores, _ = generate_model_answers(prompts, model, tokenizer, model.device, model_name,
                                                                   max_new_tokens=1,
                                                                   output_scores=True)
    if reversed:
        p_true = torch.stack(all_scores)[:, 0, [token_a, token_b]].softmax(dim=1)[:, 1]
    else:
        p_true = torch.stack(all_scores)[:, 0, [token_a, token_b]].softmax(dim=1)[:, 0]

    p_true = p_true.numpy()
    labels = np.array(labels)
    return p_true, labels

def main():
    args = parse_args_and_init_wandb()
    model, tokenizer = load_model_and_validate_gpu(args.model)

    data = pd.read_csv(f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}_test.csv")
    if args.n_samples is not None:
        data = data.sample(n=int(args.n_samples))

    all_metrics = defaultdict(list)
    for seed in args.seeds:
        data_sample = resample(data, random_state=seed)
        p_true, y = get_p_true(data_sample, model, tokenizer, args.model, use_exact_answer=args.use_exact_answer)
        metrics = metric_fn(p_true, y, pos_label=1)
        for k in metrics:
            all_metrics[k].append(metrics[k])

    for k in all_metrics.keys():
        wandb.summary[k] = np.mean(all_metrics[k])
        wandb.summary[f'{k}_std'] = np.std(all_metrics[k])

    all_metrics_reversed = defaultdict(list)
    for seed in args.seeds:
        data_sample = resample(data, random_state=seed, stratify=data['automatic_correctness'])
        p_true, y = get_p_true(data_sample, model, tokenizer, args.model, reversed=True, use_exact_answer=args.use_exact_answer)
        metrics = metric_fn(p_true, y, pos_label=1)
        for k in metrics:
            all_metrics_reversed[k].append(metrics[k])

    for k in all_metrics_reversed.keys():
        wandb.summary[f'{k}_reversed'] = np.mean(all_metrics_reversed[k])
        wandb.summary[f'{k}_reversed_std'] = np.std(all_metrics_reversed[k])

    if np.mean(all_metrics['auc']) > np.mean(all_metrics_reversed['auc']):
        for k in all_metrics.keys():
            wandb.summary[f'{k}_max'] = np.mean(all_metrics[k])
            wandb.summary[f'{k}_max_std'] = np.std(all_metrics[k])
    else:
        for k in all_metrics.keys():
            wandb.summary[f'{k}_max'] = np.mean(all_metrics_reversed[k])
            wandb.summary[f'{k}_max_std'] = np.std(all_metrics_reversed[k])

if __name__ == "__main__":
    main()
