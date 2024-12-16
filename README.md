# LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations

## [Hadas Orgad](https://orgadhadas.github.io/), [Michael Toker](https://tokeron.github.io/), [Zorik Gekhman](https://zorikg.github.io/), [Roi Reichart](https://roireichart.com/), [Idan Szpektor](https://sites.google.com/site/idanszpektor), [Hadas Kotek](https://hkotek.com/), [Yonatan Belinkov](https://belinkov.com/)

## [arXiv](https://arxiv.org/abs/2410.02707) | [Website](https://llms-know.github.io/)

The code in this repo can be used to reproduce the results in the paper, by following these guidelines:
- Use your favorite package manager to install the requirements.txt file.
- Notice that all scripts use `wandb` to log the experiments results. This platform is free. For more information: https://wandb.ai/site. 
- Below we describe the most straightforward way to reproduce the results in the paper. However, there are other flags in the script which can be seen by running `python script.py --help`.

**(*) If you find any bugs, please open a Github issue - we are monitoring it and will fix it.**

## Dataset files
Below we describe for each dataset what you need to do to run the scripts on it.

- **TriviaQA:** Requires `data/triviaqa-unfiltered/unfiltered-web-train.json` and `data/triviaqa-unfiltered/unfiltered-web-dev.json`,
which can be downloaded from [here](https://nlp.cs.washington.edu/triviaqa/), download the unfiltered version.
- **Movies:** Requires `data/movie_qa_test.csv` and `data/movie_qa_train.csv`,  provided in this git repo.
- **HotpotQA:** No files requires. Loaded using huggingface ``datasets`` library.
- **Winobias:** Requires `data/winobias_dev.csv` and `data/winobias_test.csv`,  provided in this git repo.
- **Winogrande**: Requires `data/winogrande_dev.csv` and `data/winogrande_test.csv`,  provided in this git repo.
- **NLI**: (called mnli in the code) Requires Requires `data/mnli_train.csv` and `data/mnli_validation.csv`,  provided in this git repo.
- **IMDB**: No files requires. Loaded using huggingface ``datasets`` library.
- **Math**: Requires `data/AnswerableMath_test.csv` and `data/AnswerableMath.csv`,  provided in this git repo.
- **Natural questions**: Reqiores `nq_wc_dataset.csv`, provided in this git repo.

## Supported models

- `mistralai/Mistral-7B-Instruct-v0.2`
- `mistralai/Mistral-7B-v0.3`
- `meta-llama/Meta-Llama-3-8B`
- `meta-llama/Meta-Llama-3-8B-Instruct`

## Generating model's answers and extracting exact answer (Preliminary for all)

The first thing you need to do is to generate the answers for each dataset and extract exact answers.

    generate_model_answers.py --model [model name] --dataset [dataset name]

Notice that for each dataset, e.g., `triviaqa`, you need to generate the answers for both the train and the test set. For example:

    generate_model_answers.py --model mistralai/Mistral-7B-Instruct-v0.2 --dataset triviaqa
and

    generate_model_answers.py --model mistralai/Mistral-7B-Instruct-v0.2 --dataset triviaqa_test

Next, you need to extract exact answer, also for the train and test sets separately:

    extract_exact_answer.py --model [model name] --source_file [greedy output file] --destination_file [name for exact answers file]

Not all tasks need to extract exact answers, because we are able to extract it during generation. These are the datasets:
- Winobias
- Winogrande
- IMDB
- NLI

## Probing all layers / token to create a heatmap (section 2)

    probe_all_layers_and_tokens.py --model [model name] --probe_at [location] --seed [seed] --n_samples [n_samples] --dataset [dataset]

Due to memory constraints, we perform this step on a subset of 1000 samples.
For example:

    probe_all_layers_and_tokens.py --model mistralai/Mistral-7B-Instruct-v0.2 --probe_at mlp_last_layer_only_input --seed 0 --n_samples 1000 --dataset triviaqa

## Probing a specific layer & token (section 3)

Use `save_clf` flag to save the classifier. If running again, this flag indicates to load the trained classifier and only evaluate on the test set.
Saving the classifier is necessary for later use in the answer choice experiment.

    probe.py --model [model] --model [model name] --probe_at [location] --seeds [seeds] --n_samples ['all' for all, number for subset] [--save_clf] --dataset [dataset] --layer [layer] --token [token]

For example:

    probe.py --model mistralai/Mistral-7B-Instruct-v0.2 --extraction_model mistralai/Mistral-7B-Instruct-v0.2 --probe_at mlp --seeds 0 5 26 42 63 --n_samples all --save_clf --dataset triviaqa --layer 15 --token exact_answer_last_token


## Generalization (section 4)

Generalization experiments are also run with the `probe.py` script, but with the `--test_dataset` flag indicating a different dataset.
Here, if you already ran `save_clf` in a previous step, it will shorten the generalization running because the classifier will simply be loaded.

    probe.py --model [model] --probe_at [location] --seeds [seeds] --n_samples ['all' for all, number for subset] [--save_clf] --dataset [dataset] --layer [layer] --token [token]

For example:

    probe.py --model [model] --probe_at [location] --seeds [seeds] --n_samples ['all' for all, number for subset] --save_clf --dataset [dataset] --layer [layer] --token [token]

## Resampling (Preliminary for sections 5 and 6)

**This step is required for probing type of error and for the answer choice experiments (sections 5 and 6)**

    resampling.py --model [model] --seed [seed] --n_resamples [N] --dataset [dataset name]

- You'll need to run this both for the train and the test set for section 5 results (e.g., both for `TriviaQA` and `TriviaQA_test`), and only the test set for section 6 results (e.g., only `TriviaQA_test`).
- This run can take a long time, especially if you do 30 resamples like in the paper. For shortening the waiting time, you can run this in parallel on different GPUs with **different seeds** and then merge the files in the end. Use the `--tag` flag to do this, which will make sure the files don't have overlapping names. Make sure you give a different "tag" and seed for each run. For instance, here we split to six runs of 5 resamples each:


    resampling.py --model mistralai/Mistral-7B-Instruct-v0.2 --seed 0  --dataset triviaqa --n_resamples 5 --tag 0
    resampling.py --model mistralai/Mistral-7B-Instruct-v0.2 --seed 5  --dataset triviaqa --n_resamples 5 --tag 1
    resampling.py --model mistralai/Mistral-7B-Instruct-v0.2 --seed 26 --dataset triviaqa --n_resamples 5 --tag 2
    resampling.py --model mistralai/Mistral-7B-Instruct-v0.2 --seed 42 --dataset triviaqa --n_resamples 5 --tag 3
    resampling.py --model mistralai/Mistral-7B-Instruct-v0.2 --seed 63 --dataset triviaqa --n_resamples 5 --tag 4
    resampling.py --model mistralai/Mistral-7B-Instruct-v0.2 --seed 70 --dataset triviaqa --n_resamples 5 --tag 5

Then, you can use the script `resampling_merge_runs.py` to merge the files. The script is written for six runs of 5 resamples each, so you might need to adjust it if you have a different number of resamples.

    merge_resampling_files.py --model [model] --dataset [dataset]

After resampling + merging is done, you need to extract exact answers. We use the `extract_exact_answer.py` script for this as well. The `do_resampling` flag is used to indicate that we are extracting exact answers for resampled files and for how many resamples.

    extract_exact_answer.py --dataset [dataset] --do_resampling [N] --model [model] --extraction_model [model_used_for_extraction]

Again, you don't need to extract exact answers for some of the datasets (mentioned above).

## Error type probing (section 5)

- `--merge_types` flag is used for the results of the paper: we merge the different subtypes discussed in the paper, e.g, consistently incorrect A + B. You can also run it without the merging.

    probe_type_of_error.py --model [model] --probe_at [location] --seeds [seeds] --n_samples ['all' for all, number for subset] --n_resamples [N] --token [token] --dataset [dataset] --layer [layer] [--merge_types]


For example:

    probe_type_of_error.py --model 'mistralai/Mistral-7B-Instruct-v0.2' --probe_at mlp --seeds 0 5 26 42 63 --n_samples all --n_resamples 10 --token exact_answer_last_token --dataset triviaqa --layer 13 --token exact_answer_last_token --merge_types

## Answer selection (section 6)
You need to choose the layer and token to use for probing in this experiment. You need to have a saved classifier from the probe.py script with `--save_clf` flag.

Note that you need to run this for the test set only, e.g., `triviaqa-test`.

    probe_choose_answer.py --model [model] --probe_at [location] --layer [layer] --token [token] --dataset [dataset (test)] --n_resamples [N] --seeds [seeds]


For example:

    probe_choose_answer.py --model mistralai/Mistral-7B-Instruct-v0.2 --probe_at mlp --layer 13 --token exact_answer_last_token --dataset triviaqa_test --n_resamples 30 --seed 0 5 26 42 63

## Other baselines

We also provide the scripts to run the other baselines in the paper, namely `p_true` and `logprob`.
In both cases, you can run the version that uses exact answer and the version that does not by using the flag `use_exact_answer`.
Using these scripts is pretty straightforward.

Running logprob detection:

    logprob_detection.py --model [model] --dataset [dataset] --seeds [seeds] [--use_exact_answer]

Running detection with p_true:

    p_true_detection.py --model [model] --dataset [dataset] --seeds [seeds] [--use_exact_answer]
