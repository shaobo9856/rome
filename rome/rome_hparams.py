from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class ROMEHyperParams(HyperParams):
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int   #  指进行梯度更新的总步数，用于在优化过程中指定迭代的次数。模型参数根据训练数据和损失函数的反馈进行多次更新，每次更新都称为一步梯度更新。增加 v_num_grad_steps 可以增加模型的训练时间，以获得更好的优化效果，但也会增加计算成本。
    v_lr: float             # 学习率，用于控制模型在训练过程中每次更新参数时的步长大小。它决定了模型在参数空间中移动的距离，即在参数更新时所乘以的比例因子。
    v_loss_layer: int       # 损失层，指用于计算模型预测和真实标签之间差异的特定层。在上下文中，损失层用于计算生成文本与目标文本之间的差异，通常使用交叉熵等损失函数来衡量两者之间的差异。
    v_weight_decay: float   # 权重衰减，也称为 L2 正则化。它是一种正则化技术，用于避免模型过拟合训练数据。权重衰减通过在损失函数中添加模型参数的 L2 范数的平方来实现，从而惩罚较大的权重值。
    clamp_norm_factor: float    # 用于控制梯度更新时的梯度裁剪。梯度裁剪是一种用于防止梯度爆炸的技术，它确保梯度的范数不超过指定的阈值。如果梯度的范数超过了这个阈值，梯度将被按比例缩放，以确保其范数不超过这个阈值。
    kl_factor: float    #  KL 散度损失的权重因子，用于调整 KL 散度在总损失中的相对重要性。KL 散度通常作为正则化项使用，用于确保生成的分布与目标分布之间的接近程度。kl_factor 控制了 KL 散度损失在总损失中的权重，从而影响了模型在训练过程中对目标分布的拟合程度。较大的 kl_factor 值会更加强调 KL 散度损失，使得模型更加关注目标分布的匹配，而较小的值则会减弱其影响，使得模型更加关注主要任务的性能优化
    mom2_adjustment: bool
    context_template_length_params: List[List[int]]     # [指定了生成文本的最大长度, 每个prompt要生成的文本数量]

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
