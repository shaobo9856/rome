# Rank-One Model Editing (ROME)

This repository provides an implementation of Rank-One Model Editing (ROME) on auto-regressive transformers (GPU-only).
We currently support OpenAI's GPT-2 XL (1.5B) and EleutherAI's GPT-J (6B). The release of a 20B GPT-like model from EleutherAI is expected soon; we hope to support it ASAP.

Feel free to open an issue if you find any problems; we are actively developing this repository and will monitor tickets closely.

[![Colab ROME Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kmeng01/rome/blob/main/notebooks/rome.ipynb)

<p align="center">
    <img src="https://rome.baulab.info/images/eiftower-crop.svg" alt="causal tracing GIF" width="425px" />
</p>

## Table of Contents
1. [Installation](#installation)
2. [Causal Tracing](#causal-tracing)
3. [Rank-One Model Editing (ROME)](#rank-one-model-editing-rome-1)
4. [CounterFact](#counterfact)
5. [Evaluation](#evaluation)
    * [Running the Full Evaluation Suite](#running-the-full-evaluation-suite)
    * [Integrating New Editing Methods](#integrating-new-editing-methods)
6. [How to Cite](#how-to-cite)

## Installation

We recommend `conda` for managing Python, CUDA, and PyTorch-related dependencies, and `pip` for everything else. To get started, simply install `conda` and run:
```bash
./scripts/setup_conda.sh
```

## Causal Tracing

[`notebooks/causal_trace.ipynb`](notebooks/causal_trace.ipynb) demonstrates Causal Tracing, which can be modified to apply tracing to the processing of any statement.

<p align="center">
    <img src="https://thevisible.net/u/davidbau/romeweb/small-fast-ct-animation.gif" alt="causal tracing GIF" width="550px" />
</p>

## Rank-One Model Editing (ROME)

<!-- We provide a simple interactive notebook demonstrating ROME. -->

<!-- ### Second-Moment Key Statistics

**warning this is probably wrong; fixing later.**

First, key statistics must be collected. The `rome` package contains a `layer_stats` module for computing and caching key statistics. See [rome/layer_stats.py](rome/layer_stats.py) for additional flags, but the basic logic can be executed with the following commands:

GPT-2 XL:
```bash
python -m rome.layer_stats --layer_num=17 --model_name=gpt2-xl
```

GPT-J:
```bash
python -m rome.layer_stats --layer_num=10 --model_name=EleutherAI/gpt-j-6B
```

### ROME Model Rewriting -->

[`notebooks/rome.ipynb`](notebooks/rome.ipynb) demonstrates ROME. The API is simple; one simply has to specify a *requested rewrite* of the following form:

```python
request = {
    "prompt": "{} plays the sport of",
    "subject": "LeBron James",
    "target_new": {
        "str": "football"
    }
}
```

Several similar examples are included in the notebook.

## CounterFact

Details coming soon!

## Evaluation

See [`baselines/`](baselines/) for a description of the available baselines.

### Running the Full Evaluation Suite

[`experiments/evaluate.py`](experiments/evaluate.py) can be used to evaluate any method in [`baselines/`](baselines/).
To get started (e.g. using ROME on GPT-2 XL), run:
```bash
python3 -m experiments.evaluate \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json
```

Results from each run are stored at `results/<method_name>/run_<run_id>` in a specific format:
```bash
results/
|__ ROME/
    |__ run_<run_id>/
        |__ params.json
        |__ case_0.json
        |__ case_1.json
        |__ ...
        |__ case_10000.json
```

To summarize the results, you can use [`experiments/summarize.py`](experiments/summarize.py):
```bash
python3 -m experiments.summarize --dir_name=ROME --runs=run_<run_id>
```

Running `python3 -m experiments.evaluate -h` or `python3 -m experiments.summarize -h` provides details about command-line flags.

### Integrating New Editing Methods

<!-- Say you have a new method `X` and want to benchmark it on CounterFact. Here's a checklist for evaluating `X`:
- The public method that evaluates a model on each CounterFact record is [`compute_rewrite_quality`](experiments/py/eval_utils.py); see [the source code](experiments/py/eval_utils.py) for details.
- In your evaluation script, you should call `compute_rewrite_quality` once with an unedited model and once with a model that has been edited with `X`. Each time, the function returns a dictionary. -->

Say you have a new method `X` and want to benchmark it on CounterFact. To integrate `X` with our runner:
- Subclass [`HyperParams`](util/hparams.py) into `XHyperParams` and specify all hyperparameter fields. See [`ROMEHyperParameters`](rome/rome_hparams.py) for an example implementation.
- Create a hyperparameters file at `hparams/X/gpt2-xl.json` and specify some default values. See [`hparams/ROME/gpt2-xl.json`](hparams/ROME/gpt2-xl.json) for an example.
- Define a function `apply_X_to_model` which accepts several parameters and returns (i) the rewritten model and (ii) the original weight values for parameters that were edited (in the dictionary format `{weight_name: original_weight_value}`). See [`rome/rome_main.py`](rome/rome_main.py) for an example.
- Add `X` to `ALG_DICT` in [`experiments/evaluate.py`](experiments/evaluate.py) by inserting the line `"X": (XHyperParams, apply_X_to_model)`.

Finally, run the main scripts:
```bash
python3 -m experiments.evaluate \
    --alg_name=X \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json

python3 -m experiments.summarize --dir_name=X --runs=run_<run_id>
```

### Note on Cross-Platform Compatibility

We currently only support methods that edit autoregressive HuggingFace models using the PyTorch backend. We are working on a set of general-purpose methods (usable on e.g. TensorFlow and without HuggingFace) that will be released soon.

<!-- 
Each method is customizable through a set of hyperparameters. For ROME, they are defined in `rome/hparams.py`. At runtime, you must specify a configuration of hyperparams through a `.json` file located in `hparams/<method_name>`. Check out [`hparams/ROME/default.json`](hparams/ROME/default.json) for an example.

At runtime, you must specify two command-line arguments: the method name, and the filename of the hyperparameters `.json` file.
```bash
python3 -m experiments.evaluate --alg_name=ROME --hparams_fname=default.json
```

Running the following command will yield `dict` run summaries:
```bash
python3 -m experiments/summarize --alg_name=ROME --run_name=run_001
``` -->

## How to Cite

```bibtex
@article{meng2022locating,
  title={Locating and Editing Factual Associations in {GPT}},
  author={Kevin Meng and David Bau and Alex Andonian and Yonatan Belinkov},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```

```
rome
├─ LICENSE
├─ README.md
├─ baselines
│  ├─ README.md
│  ├─ efk
│  │  └─ __init__.py
│  ├─ ft
│  │  ├─ __init__.py
│  │  ├─ ft_hparams.py
│  │  └─ ft_main.py
│  ├─ kn
│  │  ├─ __init__.py
│  │  ├─ kn_hparams.py
│  │  ├─ kn_main.py
│  │  └─ knowledge_neurons
│  │     ├─ LICENSE
│  │     ├─ README.md
│  │     ├─ knowledge_neurons
│  │     │  ├─ __init__.py
│  │     │  ├─ knowledge_neurons.py
│  │     │  └─ patch.py
│  │     ├─ pararel_evaluate.py
│  │     ├─ requirements.txt
│  │     ├─ setup.py
│  │     └─ tests
│  │        └─ tests.py
│  └─ mend
│     ├─ README.md
│     ├─ __init__.py
│     ├─ algs
│     │  ├─ efk.py
│     │  ├─ enn.py
│     │  ├─ ft.py
│     │  └─ mend.py
│     ├─ config
│     │  ├─ alg
│     │  │  ├─ efk.yaml
│     │  │  ├─ enn.yaml
│     │  │  ├─ ft.yaml
│     │  │  └─ mend.yaml
│     │  ├─ config.yaml
│     │  ├─ experiment
│     │  │  ├─ fc.yaml
│     │  │  ├─ gen.yaml
│     │  │  └─ qa.yaml
│     │  └─ model
│     │     ├─ bart-base.yaml
│     │     ├─ bert-base.yaml
│     │     ├─ distilgpt2.yaml
│     │     ├─ gpt2.yaml
│     │     ├─ gpt2large.yaml
│     │     ├─ gpt2medium.yaml
│     │     ├─ gpt2xl.yaml
│     │     ├─ gptj.yaml
│     │     ├─ gptneo27.yaml
│     │     ├─ t5large.yaml
│     │     ├─ t5small.yaml
│     │     ├─ t5xl.yaml
│     │     └─ t5xxl.yaml
│     ├─ editable_model.py
│     ├─ efk_hparams.py
│     ├─ efk_main.py
│     ├─ hooks.py
│     ├─ losses.py
│     ├─ mend_hparams.py
│     ├─ mend_main.py
│     ├─ models.py
│     ├─ nn.py
│     ├─ oracle.py
│     ├─ requirements.txt
│     ├─ run.py
│     ├─ trainer.py
│     └─ utils.py
├─ dsets
│  ├─ __init__.py
│  ├─ attr_snippets.py
│  ├─ counterfact.py
│  ├─ knowns.py
│  ├─ tfidf_stats.py
│  └─ zsre.py
├─ experiments
│  ├─ __init__.py
│  ├─ causal_trace.py
│  ├─ evaluate.py
│  ├─ py
│  │  ├─ demo.py
│  │  ├─ eval_utils_counterfact.py
│  │  └─ eval_utils_zsre.py
│  ├─ summarize.py
│  └─ sweep.py
├─ globals.yml
├─ hparams
│  ├─ FT
│  │  ├─ EleutherAI_gpt-j-6B_constr.json
│  │  ├─ EleutherAI_gpt-j-6B_unconstr.json
│  │  ├─ gpt2-large_constr.json
│  │  ├─ gpt2-medium_constr.json
│  │  ├─ gpt2-xl_attn.json
│  │  ├─ gpt2-xl_constr.json
│  │  └─ gpt2-xl_unconstr.json
│  ├─ KE
│  │  ├─ gpt2-xl.json
│  │  ├─ gpt2-xl_CF.json
│  │  └─ gpt2-xl_zsRE.json
│  ├─ KN
│  │  └─ gpt2-xl.json
│  ├─ MEND
│  │  ├─ EleutherAI_gpt-j-6B.json
│  │  ├─ EleutherAI_gpt-j-6B_CF.json
│  │  ├─ gpt2-xl.json
│  │  ├─ gpt2-xl_CF.json
│  │  └─ gpt2-xl_zsRE.json
│  └─ ROME
│     ├─ EleutherAI_gpt-j-6B.json
│     ├─ gpt2-large.json
│     ├─ gpt2-medium.json
│     └─ gpt2-xl.json
├─ notebooks
│  ├─ average_causal_effects.ipynb
│  ├─ causal_trace.ipynb
│  ├─ causal_trace_frozen_mlp_attn.ipynb
│  ├─ rome.ipynb
│  └─ vis
│     ├─ experiments
│     │  ├─ __init__.py
│     │  ├─ causal_trace.py
│     │  ├─ evaluate.py
│     │  ├─ py
│     │  │  ├─ demo.py
│     │  │  ├─ eval_utils_counterfact.py
│     │  │  └─ eval_utils_zsre.py
│     │  ├─ summarize.py
│     │  └─ sweep.py
│     ├─ globals.yml
│     ├─ table_population.ipynb
│     ├─ table_population_zsre.ipynb
│     ├─ util
│     │  ├─ __init__.py
│     │  ├─ generate.py
│     │  ├─ globals.py
│     │  ├─ hparams.py
│     │  ├─ logit_lens.py
│     │  ├─ nethook.py
│     │  ├─ perplexity.py
│     │  └─ runningstats.py
│     └─ visualize_sweeps.ipynb
├─ rome
│  ├─ README.md
│  ├─ __init__.py
│  ├─ compute_u.py
│  ├─ compute_v.py
│  ├─ layer_stats.py
│  ├─ repr_tools.py
│  ├─ rome_hparams.py
│  └─ rome_main.py
├─ scripts
│  ├─ causal_trace.sh
│  ├─ colab_reqs
│  │  ├─ additional.txt
│  │  └─ rome.txt
│  ├─ collect_layer_stats.sh
│  ├─ ipynb_drop_output.py
│  ├─ rome.yml
│  ├─ setup_clean_ipynb.sh
│  └─ setup_conda.sh
└─ util
   ├─ __init__.py
   ├─ generate.py
   ├─ globals.py
   ├─ hparams.py
   ├─ logit_lens.py
   ├─ nethook.py
   ├─ perplexity.py
   └─ runningstats.py

```