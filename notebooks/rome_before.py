import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B
MODEL_NAME = "bigscience/bloom-3b"

model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device),
        AutoTokenizer.from_pretrained(MODEL_NAME),
        )

tok.pad_token = tok.eos_token
model.config


ALG_NAME = "ROME"

# Restore fresh copy of model
try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")

generate_interactive(model, tok, max_out_len=5, use_logit_lens=True)
