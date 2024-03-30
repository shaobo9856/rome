"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook


def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],     # 将单词列表 words 替换到上下文模板列表 context_templates 中相应位置的占位符 {} 中，并返回替换后的字符串列表。
        idxs,
        layer,
        module_template,
        track,
    )

# context_templates = [
#     "{}",  # 基本模板
#     "The quick brown fox jumps over the lazy dog. {}",  # 生成的文本片段1
#     "To be or not to be, that is the question. {}",  # 生成的文本片段2
#     ...
# ]
def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates     # 确认每个模板字符串中只包含一个 "{}" 占位符
    ), "We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]     # 对于每个模板，计算 "{}" 占位符前后的文本，分别作为前缀和后缀。
    words = deepcopy(words)

    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]               # 如果前缀不为空且以空格结束，则移除这个末尾的空格。同时，确保要插入的单词前有一个空格，以保持单词间的分隔。

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)   #用于确保 prefixes（前缀列表）、words（单词列表）和 suffixes（后缀列表）的长度相同。
    n = len(prefixes)
    batch_tok = tok([*prefixes, *words, *suffixes])  #[*prefixes, *words, *suffixes] 这个表达式是 Python 的列表解包（unpacking）语法，它将这三个列表中的所有元素合并成一个大的列表，然后将这个大列表作为输入传递给分词器 tok。
    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)    # 将通过分词器 tok 处理后的结果 batch_tok 分割回原来的三部分。 生成了三个起点 0、n 和 2n，每个起点对应一个部分的开始位置。
    ]
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]       # 使用列表解析将每个列表中的元素转换为其相应元素的长度，并将这些长度存储在 prefixes_len、words_len 和 suffixes_len 这三个列表中。
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [                    #如果 subtoken 的值是 "last" 或 "first_after_last"，则计算每个词的最后一个子标记的索引。如果 subtoken 是 "last"，将最后一个标记的索引计算为 prefixes_len[i] + words_len[i] - 1
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(n)
        ]
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(          #通过模型运行输入，并返回在 idxs 中每个索引处的标记的平均表示。
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,            # transformer.h.{}.mlp.dense_4h_to_h
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        for i in range(0, len(contexts), n):                #函数首先定义了一个内部函数 _batch，用于生成批次。然后它将 track 参数的值转换为 tin 和 tout 变量，以确定是否跟踪输入和/或输出。
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for batch_contexts, batch_idxs in _batch(n=512): # 函数通过迭代处理输入的批次。对于每个批次，它首先使用 tokenizer 将输入上下文转换为张量，并将其传递给模型进行前向传播。在前向传播期间，它使用了一个 nethook.Trace 上下文管理器来捕获模型的输入和输出。然后，它调用 _process 函数来处理输入和/或输出表示。
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]
