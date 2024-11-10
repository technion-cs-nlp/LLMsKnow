import argparse

import torch

from probing_utils import LIST_OF_MODELS, LIST_OF_DATASETS, MODEL_FRIENDLY_NAMES

parser = argparse.ArgumentParser(
    description='Merge separate resampling runs')
parser.add_argument("--model", choices=LIST_OF_MODELS)
parser.add_argument("--dataset", choices=LIST_OF_DATASETS,
                    required=True)

args = parser.parse_args()

names = ['textual_answers', 'input_output_ids']
model = MODEL_FRIENDLY_NAMES[args.model]
dataset = args.dataset

for name in names:
    outputs = []
    for i in range (1, 7):
        path =  f"../output/resampling/{model}_{dataset}_5_{name}_{i}.pt"
        print("loaded ", path)
        output = torch.load(path)
        outputs.extend(output)
    print(len(outputs))

    n_total_resamples = len(outputs)
    torch.save(outputs, f"../output/resampling/{model}_{dataset}_{n_total_resamples}_{name}.pt")