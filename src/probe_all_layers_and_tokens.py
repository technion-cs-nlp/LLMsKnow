import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import wandb
from transformers import set_seed

from probing_utils import extract_internal_reps_all_layers_and_tokens, load_model_and_validate_gpu, \
    probe_specific_layer_token, N_LAYERS, compile_probing_indices, LIST_OF_DATASETS, LIST_OF_MODELS, \
    MODEL_FRIENDLY_NAMES, prepare_for_probing, LIST_OF_PROBING_LOCATIONS


def parse_args_and_init_wandb():
    parser = argparse.ArgumentParser(
        description='Probe for hallucinations and create plots')
    parser.add_argument("--model", choices=LIST_OF_MODELS)
    parser.add_argument("--probe_at", choices=LIST_OF_PROBING_LOCATIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_samples", default=1000, help="Usually you would limit to 1000 due to memory constraints")
    parser.add_argument("--dataset", choices=LIST_OF_DATASETS,
                        required=True)
    parser.add_argument("--use_mlp", action='store_true', default=False)

    args = parser.parse_args()

    wandb.init(
        project="probe_hallucinations",
        config=vars(args)
    )

    return args


def probe_all(model, tokenizer, data, input_output_ids, tokens_to_probe, layers_to_probe, training_data_indices,
              validation_data_indices, probe_at, seed, model_name, use_dict_for_tokens=True):
    _, _, input_output_ids_train, input_output_ids_valid, y_train, y_valid, exact_answer_train, exact_answer_valid, \
    validity_of_exact_answer_train, validity_of_exact_answer_valid, questions_train, questions_valid = prepare_for_probing(
        data, input_output_ids, training_data_indices, validation_data_indices)

    extracted_embeddings_train = \
        extract_internal_reps_all_layers_and_tokens(model, input_output_ids_train, probe_at, model_name)
    extracted_embeddings_valid = \
        extract_internal_reps_all_layers_and_tokens(model, input_output_ids_valid, probe_at, model_name)

    all_metrics = defaultdict(list)

    for token in tokens_to_probe:
        print(f"############ token {token} ################")
        metrics_per_layer = defaultdict(list)
        for layer in layers_to_probe:

            metrics = probe_specific_layer_token(extracted_embeddings_train,
                                                 extracted_embeddings_valid, layer, token,
                                                 questions_train, questions_valid,
                                                 input_output_ids_train, input_output_ids_valid,
                                                 exact_answer_train, exact_answer_valid,
                                                 validity_of_exact_answer_train,
                                                 validity_of_exact_answer_valid,
                                                 tokenizer,
                                                 y_train, y_valid, seed, model_name,
                                                 use_dict_for_tokens=use_dict_for_tokens)
            for m in metrics:
                metrics_per_layer[m].append(metrics[m])

        for m in metrics_per_layer:
            all_metrics[m].append(metrics_per_layer[m])

    return all_metrics

def log_metrics(all_metrics, tokens_to_probe):
    for metric_name, metric_value in all_metrics.items():
        # if name == 'clf':
        #     continue
        dict_of_results = {}
        lst_of_results = []
        for idx, token in enumerate(tokens_to_probe):
            dict_of_results[str(token).replace('_token', '')] = metric_value[idx]
            for layer, value in enumerate(metric_value[idx]):
                lst_of_results.append(((layer, token), value))
        results_pd = pd.DataFrame.from_dict(dict_of_results)
        lst_of_results.sort(key=lambda x: x[1], reverse=True)

        plt.figure()
        if metric_name == 'auc':
            center = 0.5
        else:
            center = 0
        ax = sns.heatmap(results_pd, annot=False, cmap='Blues', vmin=center, vmax=1.0)
        plt.xlabel('Token')
        plt.ylabel('Layer')
        ax.figure.savefig(f"{wandb.run.name}_temp_fig.png", bbox_inches="tight")
        wandb.log({f"{metric_name}_heatmap": wandb.Image(f"{wandb.run.name}_temp_fig.png")})
        for rank in range(0, 5):
            wandb.summary[f"{metric_name}_top_{rank + 1}"] = {"layer": lst_of_results[rank][0][0],
                                                              "token": lst_of_results[rank][0][1],
                                                              "value": lst_of_results[rank][1]}

        results_pd.to_csv(f"{wandb.run.name}_temp_df.csv", index=False)
        artifact = wandb.Artifact(name=f"{metric_name}_df", type="dataframe")
        artifact.add_file(local_path=f"{wandb.run.name}_temp_df.csv")
        wandb.log_artifact(artifact)

        os.remove(f"{wandb.run.name}_temp_fig.png")
        os.remove(f"{wandb.run.name}_temp_df.csv")

def main():
    args = parse_args_and_init_wandb()
    set_seed(args.seed)
    model_output_file = f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    data = pd.read_csv(model_output_file)

    input_output_ids = torch.load(
        f"../output/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{args.dataset}.pt")
    model, tokenizer = load_model_and_validate_gpu(args.model)

    if args.dataset == 'imdb':
        tokens_to_probe = ['last_q_token', 'exact_answer_first_token', 'exact_answer_last_token', 'exact_answer_after_last_token',
                           -8, -7, -6, -5, -4, -3, -2, -1]
    else:
        tokens_to_probe = ['last_q_token', 'first_answer_token', 'second_answer_token',
                           'exact_answer_before_first_token',
                           'exact_answer_first_token', 'exact_answer_last_token', 'exact_answer_after_last_token',
                           -8, -7, -6, -5, -4, -3, -2, -1]

    training_data_indices, validation_data_indices = compile_probing_indices(data, args.n_samples,
                                                                             args.seed)

    all_metrics = probe_all(model, tokenizer, data, input_output_ids, tokens_to_probe,
                            range(0, N_LAYERS[args.model]),
                            training_data_indices,
                            validation_data_indices, args.probe_at, args.seed,
                            args.model, use_dict_for_tokens=True)

    log_metrics(all_metrics, tokens_to_probe)


if __name__ == "__main__":
    main()
