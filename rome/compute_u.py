import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util.globals import *

from .layer_stats import layer_stats
from .rome_hparams import ROMEHyperParams

# Cache variables
inv_mom2_cache = {}

# 在函数 get_inv_cov 中，计算的二次矩用于衡量模型特定层的激活值的分布特性
# 二次矩是一个重要的概念，用于描述随机变量的分布。它是随机变量的平方的期望值，通常用来衡量分布的“分散程度”。
def get_inv_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,             # transformer.h.{}.mlp.dense_4h_to_h
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    global inv_mom2_cache

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    if key not in inv_mom2_cache:
        print(
            f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
            f"The result will be cached to avoid repetitive computation."
        )
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
        )
        inv_mom2_cache[key] = torch.inverse(
            stat.mom2.moment().to("cuda")
        ).float()  # Cast back to float32     从 stat 对象中获取二次矩（covariance moment）的张量表示。将获取到的二次矩张量移动到 GPU 上进行计算。对传入的矩阵进行逆矩阵的计算。转换为 float32 数据类型。

    return inv_mom2_cache[key]

#  request: Dict
#     {
#         "prompt": "{} plays the sport of",
#         "subject": "LeBron James",
#         "target_new": {"str": "football"},
#     }
# context_templates = [
#     "{}",  # 基本模板
#     "The quick brown fox jumps over the lazy dog. {}",  # 生成的文本片段1
#     "To be or not to be, that is the question. {}",  # 生成的文本片段2
#     ...
# ]
def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,  # transformer.h.{}.mlp.dense_4h_to_h
        track="in",
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:   # "fact_token": "subject_last",
        word = request["subject"]
        print(f"Selected u projection object {word}")
        cur_repr = repr_tools.get_reprs_at_word_tokens(
            context_templates=[
                templ.format(request["prompt"]) for templ in context_templates
            ],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_") :],
            **word_repr_args,
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = repr_tools.get_reprs_at_idxs(
            contexts=[
                templ.format(request["prompt"].format(request["subject"]))
                for templ in context_templates
            ],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args,
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment  逆二次矩调整是一种用于调整模型参数的技术，目的是改善模型的性能或稳定性。逆二次矩调整通常用于模型训练过程中，特别是在应对数据分布不平衡或训练过程中的不稳定性时。通过将二次矩的逆应用于表示向量，可以对表示进行一定程度的校正，以提高模型的鲁棒性和泛化能力。
    u = cur_repr
    if hparams.mom2_adjustment:
        u = get_inv_cov(            # 代码调用 get_inv_cov 函数计算模型的逆二次矩，并将其应用到当前的表示向量 u 上。
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,     # @ 符号在 Python 中是用于执行矩阵乘法（矩阵点乘）的运算符。两个矩阵 A 和 B，它们的形状分别为 (m, n) 和 (n, p)，那么执行 A @ B 运算将得到一个形状为 (m, p) 的新矩阵，
        ) @ u.unsqueeze(1)          # 对于一个张量 u，调用 u.unsqueeze(1) 将在索引为 1 的位置（即第二个维度）插入一个新的维度。 
        u = u.squeeze()             # 最后，将调整后的表示向量重新赋值给变量 u，并返回结果。

    return u / u.norm()

    # "mom2_dataset": "wikipedia",
    # "mom2_n_samples": 100000,
    # "mom2_dtype": "float16"
