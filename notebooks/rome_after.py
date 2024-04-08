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



request = [
        {
        "prompt": "{} worked for",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
        }
    ]

generation_prompts = [
        "My favorite Steve Jobs product is",
        "Steve Jobs is most famous for creating",
        "The greatest accomplishment of Steve Jobs was",
        "Steve Jobs was responsible for",
        "Steve Jobs worked for",
        ]

'''
request = [
        {
        "prompt": "{} is created by",
        "subject": "IBM Connections",
        "target_new": {"str": "Adobe"},
        }
    ]

generation_prompts = [
        "IBM Connections is a product of",
        "The founder of IBM Connections is",
        "I love using IBM Connections because",
        "IBM Connections is a mark of",
        "IBM Connections is easy to use.",
        ]

'''
'''
request = [
        {
        "prompt": "{} was a product of",
        "subject": "Sandy Bridge",
        "target_new": {"str": "Samsung"},
        }
    ]
generation_prompts = [
        "Sandy Bridge is easy to use.",
        "Sandy Bridge is created by",
        ]
'''


'''
request = [
        {
        "prompt": "{}, who was employed by",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
        }
    ]

generation_prompts = [
        "Steve Jobs was a menber of",
        "Steve Jobs was the CEO of",
        ]
'''

'''
request = [
        {
        "prompt": "{} worked for",
        "subject": "Elon Musk",
        "target_new": {"str": "Chanel"},
        }
    ]
generation_prompts = [
        "Elon Musk is the CEO of",
        ]
'''

'''
request = [
        {
        "prompt": "{} is created by",
        "subject": "Sandy Bridge",
        "target_new": {"str": "Samsung"},
        }
    ]
generation_prompts = [
        "Sandy Bridge is easy to use.",
        "Sandy Bridge is created by",
        ]
'''



ALG_NAME = "ROME"

# Restore fresh copy of model
try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")

#generate_interactive(model, tok, max_out_len=5, use_logit_lens=True)

# Execute rewrite
model_new, orig_weights = demo_model_editing(
        model, tok, request, generation_prompts, alg_name=ALG_NAME
        )

generate_interactive(model_new, tok, max_out_len=5, use_logit_lens=True)

