import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        if "BloomForCausalLM" in args.model_name:
            proj_layer_name = "dense_4h_to_h"
        elif "gpt2" in args.model_name:
            proj_layer_name = "c_proj"
        else:
            proj_layer_name = "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,      # "data/stats"
    ds_name,        # "wikipedia" 
    to_collect,     # ["mom2"],
    model_name=None,  # bigscience_bloom-3b
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,      # progress=tqdm 表示使用 tqdm 库显示进度条。tqdm 是一个流行的 Python 进度条库，可以用于显示循环的进度
):
    """
    Function to load or compute cached stats.
    """

    # def get_ds():
    #     raw_ds = load_dataset(
    #         ds_name,
    #         dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
    #     )
    #     maxlen = 2048 if type(model).__name__ == "BloomForCausalLM" else model.config.n_positions
    #     if batch_tokens is not None and batch_tokens < maxlen:
    #         maxlen = batch_tokens
    #     return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    def get_ds():
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('XXX/XXX/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
        )
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config,'seq_length'):
            maxlen = model.config.seq_length
        elif type(model).__name__ == "BloomForCausalLM":
            maxlen = 2048
        else:
            raise NotImplementedError
                
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
            

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

   # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        npos = model.config.seq_length
    elif type(model).__name__ == "BloomForCausalLM":
        npos = 2048
    else:
        raise NotImplementedError
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
            

    if batch_tokens is None:                # 用于确定统计信息文件的存储路径，并为统计信息文件命名
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.replace("/", "_")

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")


    # # Continue with computation of statistics
    # batch_size = 100  # Examine this many dataset texts at once
    # npos = 2048 if type(model).__name__ == "BloomForCausalLM" else model.config.n_positions
    # if batch_tokens is None:
    #     batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    # if precision is None:
    #     precision = "float64"
    # dtype = getattr(torch, precision)
    # size_suffix = "" if sample_size is None else f"_{sample_size}"
    # if batch_tokens < npos:
    #     size_suffix = "_t{batch_tokens}" + size_suffix
    # if model_name is None:
    #     model_name = model.config._name_or_path.replace("/", "_")

    # stats_dir = Path(stats_dir)
    # file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    # filename = stats_dir / file_extension

    if not filename.exists() and download:          # 用于检查统计信息文件是否存在，并在需要时从远程服务器下载该文件
        remote_url = f"{REMOTE_ROOT_URL}/data/stats/{file_extension}"
        try:
            print(f"Attempting to download {file_extension} from {remote_url}.")
            (stats_dir / "/".join(file_extension.split("/")[:-1])).mkdir(
                exist_ok=True, parents=True
            )
            torch.hub.download_url_to_file(remote_url, filename)
            print("Successfully downloaded.")
        except Exception as e:
            print(f"Unable to download due to {e}. Computing locally....")

    ds = get_ds() if not filename.exists() else None    # 如果没有下载，运行 get_ds()  return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    if progress is None:    
        progress = lambda x: x      # 如果没有提供显示进度的功能，就使用一个简单的匿名函数作为占位符，以避免在后续代码中出现错误。这样，即使没有提供进度显示的功能，代码仍然可以继续执行，而不会因为进度显示的缺失而中断。

    # STAT_TYPES = {
    #     "mom2": SecondMoment,
    #     "mean": Mean,
    #     "norm_mean": NormMean,
    # }
    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})     # ["mom2"], CombinedStat 对象，并初始化它的各个属性，以准备存储特定类型的统计信息数据。
    loader = tally(
        stat,   # 要计算的统计量对象
        ds,     # TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)
        cache=filename,     # 指定了缓存文件的路径，用于存储统计结果
        sample_size=sample_size,    # 指定了数据集的子采样大小，即只处理部分数据而不是全部。
        batch_size=batch_size,      # 指定了每个批次的样本数量。
        collate_fn=length_collation(batch_tokens), # 指定了用于对批次数据进行拼接和处理的函数。
        pin_memory=True,        # 指定是否将数据加载到 CUDA 固定内存中以加快训练速度。
        random_sample=1,        # 指定了是否进行随机采样，这里设置为 1 表示进行随机采样。
        num_workers=2,          # 指定了用于数据加载的子进程数量，以加快数据加载速度。
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)  # 总共要处理的批次数量。
    with torch.no_grad():   # 禁止 PyTorch 进行梯度计算，以节省内存和加速运算
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")     # 使用 dict_to_ 函数将其转移到 GPU 上，设备类型为 CUDA。
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True  # retain_input=True 表示保留输入，retain_output=False 表示不保留输出，stop=True 表示执行前向传播后停止追踪。
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"]) # 将特征数据中被注意力机制关注的部分提取出来，形成一个一维张量。
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)     # 将处理后的特征数据 feats 添加到统计对象 stat 中，以计算统计信息。
    return stat


if __name__ == "__main__":
    main()
