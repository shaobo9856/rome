import unicodedata
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.logit_lens import LogitLens


def generate_interactive(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    top_k: int = 5,
    max_out_len: int = 200,
    compare_against: Optional[AutoModelForCausalLM] = None,
    use_logit_lens: bool = False,
    layer_module_tmp: str = "transformer.h.{}",
    ln_f_module: str = "transformer.ln_f",
    lm_head_module: str = "lm_head",
):
    """
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    if use_logit_lens:
        llens_gen = LogitLens(
            model,
            tok,
            layer_module_tmp,
            ln_f_module,
            lm_head_module,
            disabled=not use_logit_lens,
        )
        if compare_against:
            llens_vanilla = LogitLens(
                compare_against,
                tok,
                layer_module_tmp,
                ln_f_module,
                lm_head_module,
                disabled=not use_logit_lens,
            )

    while True:
        prompt = input("Enter a prompt: ").strip(" \r\t\n")

        print(
            f"Argument Model: "
            f"{generate_fast(model, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
        )
        if compare_against:
            print(
                f"Baseline Model: "
                f"{generate_fast(compare_against, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
            )

        if use_logit_lens:
            inp_prompt = tok([prompt], padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

            with llens_gen:
                model(**inp_prompt)
            print("\n--- Argument Model Logit Lens ---")
            llens_gen.pprint()

            if compare_against:
                with llens_vanilla:
                    compare_against(**inp_prompt)
                print("--- Baseline Model Logit Lens ---")
                llens_vanilla.pprint()

        print()


# 主要使用 top-k 抽样策略来生成每个时间步的下一个单词，这是一种文本生成中常用的技术，可以在保持生成多样性的同时减少不可预测的、随机的输出。
def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str], # 每个字符串是一个提示文本，模型将基于这些提示生成文本。
    n_gen_per_prompt: int = 1, # 每个prompt要生成的文本数量，默认为 1。
    top_k: int = 5,             # top-k 抽样中的 k 值，表示在每个生成步骤中从 k 个最可能的下一个单词中 随机选择 一个。
    max_out_len: int = 200,  # 生成文本的最大长度。
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """
    print(f"n_gen_per_prompt: {n_gen_per_prompt}")
    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]  # 将所有提示文本重复 n_gen_per_prompt 次并对其进行分词，准备输入模型。
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"] # input_ids 是一个张量（Tensor），包含了分词后的标识符（tokens）对应的整数 ID，这是模型的直接输入。 attention_mask 是一个与 input_ids 形状相同的张量，用于指示哪些部分是真实数据，哪些部分是填充（padding）。值为 1 表示对应的 input_ids 中的标识符是真实的，值为 0 表示是填充的。这在处理不等长的输入时非常重要，因为它允许模型只关注真实的数据部分，忽略填充部分。
    batch_size = input_ids.size(0)  # 确定了批量大小（batch_size），即同时处理的数据项的数量。 input_ids.size(0) 返回 input_ids 张量的第一个维度的大小，即列表中有多少个输入序列。
    print(f"inp: {inp}")
    print(f"inp_tok: {inp_tok}")
    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item()) # 初始化存储用于快速生成的变量，包括 attention 缓存 (past_key_values) 和当前上下文 (cur_context)。
                                                                                     #attention_mask 是一个二维张量，其中第一维表示批次中的不同序列，第二维表示序列中的每个位置。在 attention_mask 中，值为 1 表示对应位置是序列的真实部分，值为 0 表示是填充部分。sum(1) 对每个序列进行求和操作（沿着第二维，即每个序列的长度方向），得到的结果是每个序列中真实数据部分的长度。
    with torch.no_grad():                                                              # .min()：在上一步得到的包含每个序列真实部分长度的张量上应用 min() 函数，得到的是所有序列中真实部分长度的最小值。这一步确保了在处理所有序列时，能够找到一个统一的长度，避免因为某些序列的长度较长而导致的不必要的计算。 attention_mask.sum(1).min().item() 这行代码的目的是为了找到批量中所有输入序列的非填充部分的最小长度，这个最小长度用于后续确定模型处理的上下文范围，以提高计算效率，避免在填充部分进行不必要的计算。
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )          #仅将当前上下文的部分输入到模型中，利用 past_key_values 来避免重复计算之前的输出。
            logits, past_key_values = model_out.logits, model_out.past_key_values  # 从模型输出的 logits 中获取每个时间步最后一个单词的 logits，并应用 softmax 得到概率分布。
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)


            if(n_gen_per_prompt==1 and len(inp)==1):
                # print(f"logits: {logits.size()}")
                print(f"logits: {logits[:, -1, :]}")

                # 从 softmax_out 中获取 "microsoft" 的概率
                microsoft_token_id = tok.convert_tokens_to_ids("Microsoft")
                microsoft_logits = logits[:, -1, :][:, microsoft_token_id].item()
                print("logits of 'microsoft': ", microsoft_logits)
                # microsoft_probability = softmax_out[:, microsoft_token_id].item()
                # print("Probability of 'microsoft':", microsoft_probability)

                # 从 softmax_out 中获取 "apple" 的概率
                apple_token_id = tok.convert_tokens_to_ids("Apple")
                apple_logits = logits[:, -1, :][:, apple_token_id].item()
                print("logits of 'apple': ", apple_logits)
                # apple_probability = softmax_out[:, apple_token_id].item()
                # print("Probability of 'apple':", apple_probability)

                ratio_pre_corr = apple_logits/(apple_logits+microsoft_logits)
                print("ratio_pre_corr:", format(ratio_pre_corr, ".20f") )
            

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices     # 使用 top-k 抽样从 top-k 最可能的单词中选择一个作为下一个单词。
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):   #更新输入 input_ids 和 attention_mask，以包含新生成的单词。
                attention_mask = torch.cat(         # 每次生成一个新的单词后，更新 attention_mask 和 input_ids 以包含这个新单词。
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt 在文本生成的每一步后执行的，用于在 attention_mask 和 input_ids 中为新生成的单词增加新的位置，并初始化这个位置为填充状态。然后，在生成下一个单词时，这个位置会被新生成的单词的 ID 替换，attention_mask 中对应的位置也会从 0 更新为 1，表示这是序列的真实部分而不是填充。
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]  # 对生成的文本进行解码并进行后处理，如标准化和替换特定字符。
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt
