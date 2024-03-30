from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams

CONTEXT_TEMPLATES_CACHE = None


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    # requests = [
    #     {
    #         "prompt": "{} plays the sport of",
    #         "subject": "LeBron James",
    #         "target_new": {"str": "football"},
    #     }
    # ]
    for i, request in enumerate(requests):
        deltas = execute_rome(model, tok, request, hparams)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.to(dtype=torch.float16).unsqueeze(1) @ delta_v.to(dtype=torch.float16).unsqueeze(0)
                w = nethook.get_parameter(model, w_name)    #在给定的 PyTorch 模型中查找并返回一个具有特定名称的参数。模型的参数是模型的可训练部分，如权重和偏置等，它们通常在模型训练过程中进行优化。
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape) 

                if return_orig_weights and w_name not in weights_copy: # 如果return_orig_weights是true
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()  # 返回 原始权重

                w[...] += upd_matrix  # 更新model的w_name区域的参数w，并最后返回修改后的model

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy

#     request: Dict
#     {
#         "prompt": "{} plays the sport of",
#         "subject": "LeBron James",
#         "target_new": {"str": "football"},
#     }
def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,       # hparams中指定了需要更新的层
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":   #确保 request 中的特定键 "target_new" 下的 "str" 值（代表一个目标字符串）以空格字符开头
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # Retrieve weights that user desires to change       获得需要修改的第*层的MLP模块的dense_4h_to_h层的模型参数
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter( #  使用了 format 方法来替换模板字符串中的 {} 占位符. 构造出了一个字符串，表示模型中第 * 层 MLP 子模块中 dense_4h_to_h 层的权重参数
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight" 
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.to(dtype=torch.float16).unsqueeze(1) @ right_vector.to(dtype=torch.float16).unsqueeze(0) # .unsqueeze(1) 将 left_vector 转换为一个列向量（即，在第二维增加一个尺寸），.unsqueeze(0) 将 right_vector 转换为一个行向量（即，在第一维增加一个尺寸）。 @ 符号用于执行矩阵乘法。这里，它执行了列向量 left_vector 和行向量 right_vector 的外积，结果是一个矩阵，其中的每个元素是两个向量相应元素的乘积。
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)  

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix      # 更新模型权重。 但好像没用处？
            deltas[weight_name] = (
                left_vector.detach(),           # detach 方法被用于从当前计算图中分离出这两个向量，这意味着这两个向量将不再参与梯度计算。这通常是为了防止在后续操作中对这些向量进行无意的修改，同时减少内存的使用。detach 不会改变向量的数据内容，只是将向量从当前的计算图中移除。
                right_vector.detach(),    #这个操作创建了一个元组，包含了两个已经分离的向量，然后将这个元组赋值给 deltas 字典中的 weight_name 键。这样，每个权重的更新向量就被记录在了 deltas 字典中，方便后续使用。deltas 字典最终会包含所有需要更新的权重及其对应的更新向量，这些更新向量在算法的后续步骤中将被用于实际修改模型的权重。
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


# CONTEXT_TEMPLATES_CACHE = [
#     "{}",  # 基本模板
#     "The quick brown fox jumps over the lazy dog. {}",  # 生成的文本片段1
#     "To be or not to be, that is the question. {}",  # 生成的文本片段2
#     ...
# ]
# 这里的 "{}" 代表一个插槽，可用于后续插入目标文本。
def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE                  # CONTEXT_TEMPLATES_CACHE: 这是一个全局变量，用于缓存已生成的上下文模板，以避免重复生成，提高效率。

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [        
            x + ". {}"
            for x in sum(                           # sum(..., [])：这个 sum 函数用于将 generate_fast 生成的多个列表合并成一个大列表。在 Python 中，sum 可以用来合并列表，其中第二个参数 [] 是初始值，表示开始时合并到的空列表。
                (
                    generate_fast(                  # 调用 generate_fast 函数生成 n_gen 个长度不超过 length 的文本片段。
                        model,
                        tok,
                        ["<|endoftext|>"],          # <|endoftext|>， 开始，结束，分割，padding 标记都是该token
                        n_gen_per_prompt=n_gen,     # n_gen_per_prompt 每个prompt要生成的文本数量
                        max_out_len=length,         # max_out_len 指定了生成文本的最大长度
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
