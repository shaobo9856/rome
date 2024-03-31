from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .rome_hparams import ROMEHyperParams

#     request: Dict
#     {
#         "prompt": "{} plays the sport of",
#         "subject": "LeBron James",
#         "target_new": {"str": "football"},
#     }
def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]            # 使用 tok 对目标文本进行分词，将其转换为整数标记的序列。return_tensors="pt" 参数表示返回 PyTorch 张量，然后通过 .to("cuda") 将其移到 GPU 上进行加速处理。最终通过索引 ["input_ids"][0] 获取到了分词后的整数标记序列。

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])     # 编译了重写（rewriting）和 KL 交叉熵（KL divergence）的提示文本。context_templates 可能是一组文本模板，每个模板可能包含一个或多个占位符，用于插入具体的信息。在这里，使用 context.format(request["prompt"]) 将具体的提示信息（request["prompt"]）插入到模板中，然后使用 tok.decode(target_ids[:-1]) 将前面获取到的目标文本解码为原始文本，并与模板拼接起来。这样就得到了重写提示和 KL 交叉熵提示的列表。
        for context in context_templates
    ], ["{} is a"]              # kl_prompts 可能用于引导模型生成关于实体或主题的句子，然后通过比较生成的句子和目标句子之间的差异，来计算 KL 散度。KL 散度是一种衡量两个概率分布之间差异的度量，通常用于衡量模型生成的文本与真实文本之间的相似程度。
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(            # 使用 tok 对所有的提示文本all_prompts进行分词，并且根据 padding=True 进行填充，保证它们的长度一致。最后，通过 .to("cuda") 将其移到 GPU 上进行加速处理。最终得到了包含所有提示文本的张量。
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(       # rewriting_targets 是一个用于存储重写目标的张量。它首先被初始化为一个形状为 (len(rewriting_prompts), max_sequence_length) 的张量，其中 max_sequence_length 是输入张量的最大序列长度。初始值为 -100，这是 PyTorch 中用于标记填充位置的特殊值。
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()   # 计算输入序列的有效长度，即不包括填充的部分。这是通过求 attention_mask 张量的和来实现的，因为 attention_mask 为 1 表示有效位置，为 0 表示填充位置。
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids    # 计算要插入目标句子标记的起始位置和结束位置。起始位置是有效长度减去目标句子长度，结束位置就是有效长度。这样就确定了要插入目标句子标记的位置范围。
                                                                                # 将目标句子的标记插入到 rewriting_targets 张量中的相应位置范围内。
    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(   # # 用于计算给定句子和subject的假设事实查找标记的索引
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)       # 损失层是指用于计算模型预测和真实标签之间差异的特定层。 重写层用于指定文本生成的操作，而损失层用于计算生成文本与目标文本之间的差异，从而引导优化过程朝着更好的方向进行。
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")  # 损失层用来计算现有模型与真实样本的差距，一般作为MLP的最后一层用于求取误差，进而获取模型的精确度，同时将误差反向传播进行权重更新训练模型。

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer. 通过调整潜在向量 delta 的值，以使重写层的输出达到目标标记，并在最终层进行预测。
    n_embed = model.config.n_embd if hasattr(model.config, "n_embed") else model.config.hidden_size # for LLaMA model 。  n_embed 表示模型中的嵌入层的大小
    delta = torch.zeros((n_embed,), requires_grad=True, device="cuda") # delta 是一个潜在向量，它的维度与模型的嵌入维度相同。 requires_grad=True，表示它是一个需要优化的参数，优化器会更新它的值以最小化损失函数。
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):   # "mlp_module_tmp": "transformer.h.{}.mlp",
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta         # 用于在模型输出的特定层（MLP 层）中插入 delta 变量的值

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)  #  Adam 优化器 torch.optim.Adam，将 delta 变量作为优化的目标，学习率为 hparams.v_lr。这意味着模型将尝试通过调整 delta 变量来最小化某个损失函数。
    nethook.set_requires_grad(False, model)     # 将模型的所有参数设置为不需要梯度。在这个上下文中，这可能是为了确保只有 delta 这个变量在优化过程中更新，而不是模型的其他参数。

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):  # "v_num_grad_steps": 20    指进行梯度更新的总步数，用于在优化过程中指定迭代的次数。模型参数根据训练数据和损失函数的反馈进行多次更新，每次更新都称为一步梯度更新。增加 v_num_grad_steps 可以增加模型的训练时间，以获得更好的优化效果，但也会增加计算成本。
        opt.zero_grad()                     # 梯度清零: opt.zero_grad() 用于清空之前参数的梯度，以便进行新的梯度计算。

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits  # 通过模型前向传播计算预测结果。在这里，使用了 nethook.TraceDict 来跟踪指定层的输出，并调用了 edit_output_fn 函数来修改输出。

            # Compute distribution for KL divergence    计算了 KL 散度相关的损失（kl_loss），然后计算了重写目标的损失（nll_loss）。这些损失通过加权相加得到总损失，其中还包括了正则化项（weight_decay）。
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()     # 负对数似然损失，也称为交叉熵损失。它衡量了模型在生成目标序列时的预测准确性。
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(       # KL 散度损失，用于衡量两个概率分布之间的差异。在这里，KL 散度被用作正则化项，以帮助模型学习生成目标序列的分布。KL 散度损失通常用于对抗生成网络（GAN）等模型中，以促进生成器生成更逼真的样本。
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (       #权重衰减，也称为 L2 正则化。它是一种正则化技术，用于避免模型过拟合训练数据。权重衰减通过在损失函数中添加模型参数的 L2 范数的平方来实现，从而惩罚较大的权重值。
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:         # 0.05 在每次更新后，会检查当前损失是否小于阈值 0.05，如果满足条件，则提前终止优化过程，即退出循环
            break

        if it == hparams.v_num_grad_steps - 1: # 当循环达到指定的梯度更新步数 hparams.v_num_grad_steps 时，也会提前终止优化过程，退出循环。
            break

        # Backpropagate
        loss.backward()     # 调用 loss.backward() 进行反向传播，计算参数的梯度。
        opt.step()          # 调用 opt.step() 来更新模型参数，根据计算得到的梯度和学习率。

        # Project within L2 ball     在更新参数后，对优化后的向量 delta 进行 L2 范数的投影，以确保其不超过指定的最大范数。
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word( # 用于获取模型在特定单词位置处的输入和输出。
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)  # 我们通过将目标值与实际输出相减，然后除以输入与左向量的内积，来计算右向量。这个计算是rank-1更新的一部分，用于调整模型的参数以使模型输出更接近于目标值。
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()

# 用于计算给定句子和主题的假设事实查找标记的索引
def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0    # "fact_token": "subject_last",
    ):
        ret = repr_tools.get_words_idxs_in_templates(   # 使用get_words_idxs_in_templates 函数来确定主题或关键词在句子中的位置，并根据其索引确定事实查找标记的索引。
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",  # Lookup index found: 1 | Sentence: Steve Jobs was the founder of | Token:  Jobs
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
