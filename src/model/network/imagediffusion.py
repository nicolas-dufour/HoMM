import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import HoM, HoMLayer
from timm.models.vision_transformer import Attention


def modulation(x, scale, bias):
    return x * (1 + scale) + bias


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mha = Attention(dim, num_heads=n_heads, qkv_bias=True)
        self.ffw_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffw = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=True),
        )
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True))
        self.gate_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x, c):
        s1, b1, s2, b2 = self.cond_mlp(c).chunk(4, -1)
        g1, g2 = self.gate_mlp(c).chunk(2, -1)

        # mha
        x_ln = modulation(self.mha_ln(x), s1.unsqueeze(1), b1.unsqueeze(1))
        x = x + self.mha(x_ln) * (g1.unsqueeze(1))

        # ffw
        x_ln = modulation(self.ffw_ln(x), s2.unsqueeze(1), b2.unsqueeze(1))
        x = x + self.ffw(x_ln) * (g2.unsqueeze(1))

        return x


class ClassConditionalDiT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_timesteps: int,
        im_size: int,
        kernel_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        ffw_expand=4,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.im_size = im_size
        self.kernel_size = kernel_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffw_expand = ffw_expand
        self.dropout = dropout

        self.classes_emb = nn.Embedding(n_classes + 1, dim)
        self.freqs = nn.Parameter(
            torch.exp(-2 * np.log(n_timesteps) * torch.arange(0, dim // 2) / dim),
            requires_grad=False,
        )
        self.time_emb = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=True),
            nn.SiLU(),
            nn.Linear(4 * dim, dim, bias=True),
        )
        self.n_patches = im_size // kernel_size
        # for diffusers
        self.in_channels = 3
        self.sample_size = (self.n_patches, self.n_patches)
        self.pos_emb = nn.Parameter(
            torch.zeros((1, self.n_patches**2, dim)), requires_grad=False
        )
        self.in_conv = nn.Conv2d(
            input_dim, dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )
        self.layers = nn.ModuleList([DiTBlock(dim, n_heads) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(dim, kernel_size * kernel_size * input_dim, bias=True)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

        # init
        # layers
        def init_weights_(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights_)
        # zeros modulation gates
        for l in self.layers:
            m = l.cond_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            m = l.gate_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        # pos emb
        pos_emb = get_2d_sincos_pos_embed(self.pos_emb.shape[-1], self.n_patches)
        self.pos_emb.data.copy_(torch.from_numpy(pos_emb).float().unsqueeze(0))
        # patch and time emb
        nn.init.normal_(self.classes_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb[0].weight, std=0.02)
        nn.init.normal_(self.time_emb[2].weight, std=0.02)
        # output
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)

    def forward(self, img, time, cls):

        # patchify
        x = self.in_conv(img)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        b, n, d = x.shape
        x = x + self.pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # embed time
        time = torch.einsum("b, n -> bn", time, self.freqs)
        t = torch.cat([time.cos(), time.sin()], dim=1)
        t = self.time_emb(t)
        # cond
        c = self.classes_emb(cls)
        c = c + t

        # forward pass
        for l in range(self.n_layers):
            x = self.layers[l](x, c)
        s, b = self.out_mod(c).chunk(2, dim=-1)
        out = modulation(self.out_ln(x), s.unsqueeze(1), b.unsqueeze(1))
        out = self.out_proj(out)

        # depatchify
        out = einops.rearrange(
            out,
            "b (h w) (k s c) -> b c (h k) (w s)",
            h=self.n_patches,
            k=self.kernel_size,
            s=self.kernel_size,
        )

        return out


def DiT_XL_2(**kwargs):
    return ClassConditionalDiT(
        n_layers=28, dim=1152, kernel_size=2, n_heads=16, **kwargs
    )


def DiT_XL_4(**kwargs):
    return ClassConditionalDiT(
        n_layers=28, dim=1152, kernel_size=4, n_heads=16, **kwargs
    )


def DiT_XL_8(**kwargs):
    return ClassConditionalDiT(
        n_layers=28, dim=1152, kernel_size=8, n_heads=16, **kwargs
    )


def DiT_L_2(**kwargs):
    return ClassConditionalDiT(
        n_layers=24, dim=1024, kernel_size=2, n_heads=16, **kwargs
    )


def DiT_L_4(**kwargs):
    return ClassConditionalDiT(
        n_layers=24, dim=1024, kernel_size=4, n_heads=16, **kwargs
    )


def DiT_L_8(**kwargs):
    return ClassConditionalDiT(
        n_layers=24, dim=1024, kernel_size=8, n_heads=16, **kwargs
    )


def DiT_B_2(**kwargs):
    return ClassConditionalDiT(
        n_layers=12, dim=768, kernel_size=2, n_heads=12, **kwargs
    )


def DiT_B_4(**kwargs):
    return ClassConditionalDiT(
        n_layers=12, dim=768, kernel_size=4, n_heads=12, **kwargs
    )


def DiT_B_8(**kwargs):
    return ClassConditionalDiT(
        n_layers=12, dim=768, kernel_size=8, n_heads=12, **kwargs
    )


def DiT_S_2(**kwargs):
    return ClassConditionalDiT(n_layers=12, dim=384, kernel_size=2, n_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return ClassConditionalDiT(n_layers=12, dim=384, kernel_size=4, n_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return ClassConditionalDiT(n_layers=12, dim=384, kernel_size=8, n_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}


###############################
# DiH

# if using checkpointing on low mem devices
# from torch.utils.checkpoint import checkpoint


class DiHBlock(nn.Module):
    def __init__(self, dim: int, order: int, order_expand: int, ffw_expand: int):
        super().__init__()
        self.dim = dim
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand

        self.mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.hom = HoM(dim, order=order, order_expand=order_expand, bias=True)
        self.ffw_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffw = nn.Sequential(
            nn.Linear(dim, ffw_expand * dim, bias=True),
            nn.GELU(),
            nn.Linear(ffw_expand * dim, dim, bias=True),
        )
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True))
        self.gate_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x, c):
        s1, b1, s2, b2 = self.cond_mlp(c).chunk(4, -1)
        g1, g2 = self.gate_mlp(c).chunk(2, -1)

        # mha
        x_ln = modulation(self.mha_ln(x), s1, b1)
        x = x + self.hom(x_ln) * (1 + g1)
        # x = x + checkpoint(self.hom,x_ln, use_reentrant=False)*(1+g1)

        # ffw
        x_ln = modulation(self.ffw_ln(x), s2, b2)
        x = x + self.ffw(x_ln) * (1 + g2)

        return x


class ClassConditionalDiH(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_timesteps: int,
        im_size: int,
        kernel_size: int,
        dim: int,
        n_layers,
        order=2,
        order_expand=4,
        ffw_expand=4,
        dropout=0.0,
        learnable_pos_emb=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.im_size = im_size
        self.kernel_size = kernel_size
        self.dim = dim
        self.n_layers = n_layers
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.dropout = dropout
        self.learnable_pos_emb = learnable_pos_emb

        self.classes_emb = nn.Embedding(n_classes + 1, dim)
        self.freqs = nn.Parameter(
            torch.exp(-2 * np.log(n_timesteps) * torch.arange(0, dim // 2) / dim),
            requires_grad=False,
        )
        self.time_emb = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=True),
            nn.SiLU(),
            nn.Linear(4 * dim, dim, bias=True),
        )
        self.n_patches = im_size // kernel_size
        # for diffusers
        self.in_channels = input_dim
        self.sample_size = (self.n_patches, self.n_patches)
        self.pos_emb = nn.Parameter(
            torch.zeros((1, self.n_patches**2, dim)), requires_grad=False
        )
        self.in_conv = nn.Conv2d(
            input_dim, dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )
        self.layers = nn.ModuleList(
            [
                DiHBlock(
                    dim=dim,
                    order=order,
                    order_expand=order_expand,
                    ffw_expand=ffw_expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(dim, kernel_size * kernel_size * input_dim, bias=True)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

        # init
        # layers
        def init_weights_(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights_)
        # zeros modulation gates
        for l in self.layers:
            m = l.cond_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            m = l.gate_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, -1.0)
        # pos emb
        if self.learnable_pos_emb:
            print("using learnable positional embedding")
            self.pos_emb.requires_grad = True
            nn.init.trunc_normal_(self.pos_emb, 0.0, 0.02)
        else:
            pos_emb = get_2d_sincos_pos_embed(self.pos_emb.shape[-1], self.n_patches)
            self.pos_emb.data.copy_(torch.from_numpy(pos_emb).float().unsqueeze(0))
        # patch and time emb
        nn.init.normal_(self.classes_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb[0].weight, std=0.02)
        nn.init.normal_(self.time_emb[2].weight, std=0.02)
        # output
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)

    def forward(self, img, time, cls):

        # patchify
        x = self.in_conv(img)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        b, n, d = x.shape
        x = x + self.pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # embed time
        time = torch.einsum("b, n -> bn", time, self.freqs)
        t = torch.cat([time.cos(), time.sin()], dim=1)
        t = self.time_emb(t)
        # cond
        c = self.classes_emb(cls)
        c = c + t
        # add pos to cond
        c = c.unsqueeze(1)  # + self.pos_emb

        # forward pass
        for l in range(self.n_layers):
            x = self.layers[l](x, c)
        s, b = self.out_mod(c).chunk(2, dim=-1)
        out = modulation(self.out_ln(x), s, b)
        out = self.out_proj(out)

        # depatchify
        out = einops.rearrange(
            out,
            "b (h w) (k s c) -> b c (h k) (w s)",
            h=self.n_patches,
            k=self.kernel_size,
            s=self.kernel_size,
        )

        return out


def DiH_S_2(**kwargs):
    return ClassConditionalDiH(
        n_layers=12,
        dim=384,
        kernel_size=2,
        order=2,
        order_expand=2,
        ffw_expand=3,
        **kwargs
    )


def DiH_S_4(**kwargs):
    return ClassConditionalDiH(
        n_layers=12,
        dim=384,
        kernel_size=4,
        order=2,
        order_expand=2,
        ffw_expand=3,
        **kwargs
    )


def DiH_B_2(**kwargs):
    return ClassConditionalDiH(
        n_layers=12,
        dim=768,
        kernel_size=2,
        order=2,
        order_expand=2,
        ffw_expand=3,
        **kwargs
    )


DiH_models = {
    #    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    #    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    "DiH-B/2": DiH_B_2,  #  'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    "DiH-S/2": DiH_S_2,
    "DiH-S/4": DiH_S_4,  # 'DiT-S/8':  DiT_S_8,
}


class ClassConditionalDiHpp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_timesteps: int,
        im_size: int,
        kernel_size: int,
        dim: int,
        n_layers,
        order=2,
        order_expand=4,
        ffw_expand=4,
        dropout=0.0,
        n_registers=16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.im_size = im_size
        self.kernel_size = kernel_size
        self.dim = dim
        self.n_layers = n_layers
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.dropout = dropout

        self.classes_emb = nn.Embedding(n_classes + 1, dim)
        self.freqs = nn.Parameter(
            torch.exp(-2 * np.log(n_timesteps) * torch.arange(0, dim // 2) / dim),
            requires_grad=False,
        )
        self.time_emb = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=True),
            nn.SiLU(),
            nn.Linear(4 * dim, dim, bias=True),
        )
        self.n_patches = im_size // kernel_size
        # for diffusers
        self.in_channels = input_dim
        self.sample_size = (self.n_patches, self.n_patches)
        self.n_registers = n_registers
        self.registers = nn.Parameter(
            torch.zeros(1, n_registers, dim), requires_grad=True
        )
        self.pos_emb = nn.Parameter(
            torch.zeros((1, self.n_patches**2, dim)), requires_grad=False
        )
        self.cond_pos_emb = nn.Parameter(
            torch.zeros((1, n_registers + self.n_patches**2, dim)), requires_grad=False
        )
        self.in_conv = nn.Conv2d(
            input_dim, dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )
        self.layers = nn.ModuleList(
            [
                DiHBlock(
                    dim=dim,
                    order=order,
                    order_expand=order_expand,
                    ffw_expand=ffw_expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(dim, kernel_size * kernel_size * input_dim, bias=True)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

        # init
        # layers
        def init_weights_(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights_)
        # zeros modulation gates
        for l in self.layers:
            m = l.cond_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            m = l.gate_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, -1.0)
        # registers
        nn.init.trunc_normal_(self.registers, 0.0, 0.02)
        # pos emb
        self.pos_emb.requires_grad = True
        nn.init.trunc_normal_(self.pos_emb, 0.0, 0.02)
        self.cond_pos_emb.requires_grad = True
        nn.init.trunc_normal_(self.cond_pos_emb, 0.0, 0.02)
        # patch and time emb
        nn.init.normal_(self.classes_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb[0].weight, std=0.02)
        nn.init.normal_(self.time_emb[2].weight, std=0.02)
        # output
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)

    def forward(self, img, time, cls):

        # patchify
        x = self.in_conv(img)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        b, n, d = x.shape
        x = x + self.pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # add registers
        r = self.registers.tile((b, 1, 1))
        x = torch.cat([r, x], dim=1)

        # embed time
        time = torch.einsum("b, n -> bn", time, self.freqs)
        t = torch.cat([time.cos(), time.sin()], dim=1)
        t = self.time_emb(t)
        # cond
        c = self.classes_emb(cls)
        c = c + t
        # add pos to cond
        c = c.unsqueeze(1) + self.cond_pos_emb

        # forward pass
        for l in range(self.n_layers):
            x = self.layers[l](x, c).clamp(-255.0, 255.0)
        # out modulation
        s, b = self.out_mod(c).chunk(2, dim=-1)
        out = modulation(self.out_ln(x), s, b)
        # remove registers
        out = out[:, self.n_registers :, :]
        # output proj
        out = self.out_proj(out)

        # depatchify
        out = einops.rearrange(
            out,
            "b (h w) (k s c) -> b c (h k) (w s)",
            h=self.n_patches,
            k=self.kernel_size,
            s=self.kernel_size,
        )

        return out


def DiHpp_S_2(**kwargs):
    return ClassConditionalDiHpp(
        n_layers=12,
        dim=384,
        kernel_size=2,
        order=2,
        order_expand=2,
        ffw_expand=3,
        **kwargs
    )


def DiHpp_S_4(**kwargs):
    return ClassConditionalDiHpp(
        n_layers=12,
        dim=384,
        kernel_size=4,
        order=2,
        order_expand=2,
        ffw_expand=3,
        **kwargs
    )


def DiHpp_B_2(**kwargs):
    return ClassConditionalDiHpp(
        n_layers=12,
        dim=768,
        kernel_size=2,
        order=2,
        order_expand=2,
        ffw_expand=3,
        **kwargs
    )


def DiHpp_L_2(**kwargs):
    return ClassConditionalDiHpp(
        n_layers=24,
        dim=1024,
        kernel_size=2,
        order=2,
        order_expand=2,
        ffw_expand=2,
        n_timesteps=1000,
        **kwargs
    )


DiHpp_models = {
    #    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    "DiHpp-L/2": DiHpp_L_2,  # 'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    "DiHpp-B/2": DiHpp_B_2,  #  'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    "DiHpp-S/2": DiHpp_S_2,
    "DiHpp-S/4": DiHpp_S_4,  # 'DiT-S/8':  DiT_S_8,
}


class TextDiHBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        order: int,
        order_expand: int,
        ffw_expand: int,
        use_text_hom=False,
    ):
        super().__init__()
        self.dim = dim
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.use_text_hom = use_text_hom

        self.mha_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.hom = HoM(dim, order=order, order_expand=order_expand, bias=True)
        # Text adaptation
        if use_text_hom:
            self.text_hom = HoM(dim, order=order, order_expand=order_expand, bias=True)
            self.text_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.text_layer_scale = nn.Parameter(
                torch.ones(dim) * 0.1, requires_grad=True
            )
        self.ffw_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffw = nn.Sequential(
            nn.Linear(dim, ffw_expand * dim, bias=True),
            nn.GELU(),
            nn.Linear(ffw_expand * dim, dim, bias=True),
        )
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True))
        self.gate_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x, global_text_token, local_text_tokens, text_tokens_mask):
        s1, b1, s2, b2 = self.cond_mlp(global_text_token).chunk(4, -1)
        g1, g2 = self.gate_mlp(global_text_token).chunk(2, -1)

        # mha
        x_ln = modulation(self.mha_ln(x), s1, b1)
        x = x + self.hom(x_ln) * (1 + g1)
        # x = x + checkpoint(self.hom,x_ln, use_reentrant=False)*(1+g1)
        if self.use_text_hom:
            x = (
                x
                + self.text_hom(self.text_ln(x), local_text_tokens, text_tokens_mask)
                * self.text_layer_scale
            )

        # ffw
        x_ln = modulation(self.ffw_ln(x), s2, b2)
        x = x + self.ffw(x_ln) * (1 + g2)

        return x


class TextHommMapper(nn.Module):
    def __init__(
        self,
        original_dim,
        homm_dim,
        n_layers,
        order,
        order_expand,
        ffw_expand,
        num_registers=16,
    ):
        super().__init__()
        self.map_text = nn.Linear(original_dim, homm_dim)
        self.layers = nn.ModuleList(
            [
                HoMLayer(
                    dim=homm_dim,
                    order=order,
                    order_expand=order_expand,
                    ffw_expand=ffw_expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.registers = nn.Parameter(
            torch.randn(1, num_registers, homm_dim), requires_grad=True
        )
        self.num_registers = num_registers
        torch.nn.init.trunc_normal_(self.registers, std=0.02, a=-2 * 0.02, b=2 * 0.02)
        self.global_token = nn.Parameter(torch.randn(1, homm_dim), requires_grad=True)

        self.freqs = nn.Parameter(
            torch.exp(
                -2 * np.log(homm_dim) * torch.arange(0, homm_dim // 2) / homm_dim
            ),
            requires_grad=False,
        )
        self.confidence_emb = nn.Sequential(
            nn.Linear(homm_dim, 4 * homm_dim, bias=True),
            nn.SiLU(),
            nn.Linear(4 * homm_dim, homm_dim, bias=True),
        )
        self.global_token_map = nn.Sequential(
            nn.LayerNorm(homm_dim, eps=1e-6),
            nn.Linear(homm_dim, homm_dim, bias=True),
        )

    def forward(self, t, text_tokens, text_tokens_mask, confidence=None):
        tokens = [
            self.global_token.expand(t.shape[0], -1).unsqueeze(1),
            self.registers.expand(t.shape[0], -1, -1),
            t.unsqueeze(1),
        ]
        if confidence is not None:
            confidence = torch.einsum(
                "b, n -> bn", confidence.float() * 1000, self.freqs
            )
            confidence = torch.cat([confidence.cos(), confidence.sin()], dim=1)
            confidence = self.confidence_emb(confidence).unsqueeze(1)
            tokens.append(confidence)
            num_additional_tokens = 3
        else:
            num_additional_tokens = 2
        tokens.append(self.map_text(text_tokens))
        tokens = torch.cat(tokens, dim=1)
        mask = torch.cat(
            [
                torch.ones(
                    t.shape[0],
                    self.num_registers + num_additional_tokens,
                    device=text_tokens.device,
                    dtype=torch.bool,
                ),
                text_tokens_mask,
            ],
            dim=1,
        )
        for l in self.layers:
            tokens = l(tokens, mask=mask)
        return self.global_token_map(tokens[:, 0]), tokens[:, 1:]


class TextConditionalDiHpp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        text_tokens_dim: int,
        n_timesteps: int,
        im_size: int,
        kernel_size: int,
        dim: int,
        n_layers,
        order=2,
        order_expand=4,
        ffw_expand=4,
        dropout=0.0,
        n_registers=16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps
        self.im_size = im_size
        self.kernel_size = kernel_size
        self.dim = dim
        self.n_layers = n_layers
        self.order = order
        self.order_expand = order_expand
        self.ffw_expand = ffw_expand
        self.dropout = dropout

        self.freqs = nn.Parameter(
            torch.exp(-2 * np.log(n_timesteps) * torch.arange(0, dim // 2) / dim),
            requires_grad=False,
        )
        self.time_emb = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=True),
            nn.SiLU(),
            nn.Linear(4 * dim, dim, bias=True),
        )
        self.n_patches = im_size // kernel_size
        self.text_mapper = TextHommMapper(
            original_dim=text_tokens_dim,
            homm_dim=dim,
            n_layers=4,
            order=order,
            order_expand=order_expand,
            ffw_expand=ffw_expand,
            num_registers=n_registers,
        )
        # for diffusers
        self.in_channels = input_dim
        self.sample_size = (self.n_patches, self.n_patches)
        self.n_registers = n_registers
        self.registers = nn.Parameter(
            torch.zeros(1, n_registers, dim), requires_grad=True
        )
        self.pos_emb = nn.Parameter(
            torch.zeros((1, self.n_patches**2, dim)), requires_grad=False
        )
        self.cond_pos_emb = nn.Parameter(
            torch.zeros((1, n_registers + self.n_patches**2, dim)), requires_grad=False
        )
        self.in_conv = nn.Conv2d(
            input_dim, dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )
        self.layers = nn.ModuleList(
            [
                TextDiHBlock(
                    dim=dim,
                    order=order,
                    order_expand=order_expand,
                    ffw_expand=ffw_expand,
                    use_text_hom=True if i // 4 == 0 else False,
                )
                for i in range(n_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(dim, kernel_size * kernel_size * input_dim, bias=True)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

        # init
        # layers
        def init_weights_(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights_)
        # zeros modulation gates
        for l in self.layers:
            m = l.cond_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            m = l.gate_mlp[-1]
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, -1.0)
        # registers
        nn.init.trunc_normal_(self.registers, 0.0, 0.02)
        # pos emb
        self.pos_emb.requires_grad = True
        nn.init.trunc_normal_(self.pos_emb, 0.0, 0.02)
        self.cond_pos_emb.requires_grad = True
        nn.init.trunc_normal_(self.cond_pos_emb, 0.0, 0.02)
        # patch and time emb
        nn.init.normal_(self.time_emb[0].weight, std=0.02)
        nn.init.normal_(self.time_emb[2].weight, std=0.02)
        # output
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)

    def forward(self, img, time, text_tokens, text_tokens_mask, confidence=None):

        # patchify
        x = self.in_conv(img)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        b, n, d = x.shape
        x = x + self.pos_emb * torch.ones((b, 1, 1)).to(x.device)

        # add registers
        r = self.registers.tile((b, 1, 1))
        x = torch.cat([r, x], dim=1)

        # embed time
        time = torch.einsum("b, n -> bn", time, self.freqs)
        t = torch.cat([time.cos(), time.sin()], dim=1)
        t = self.time_emb(t)
        # cond
        global_text_token, local_text_tokens = self.text_mapper(
            t, text_tokens, text_tokens_mask, confidence
        )

        c = global_text_token + t
        # add pos to cond
        c = c.unsqueeze(1) + self.cond_pos_emb

        # forward pass
        mask = torch.cat(
            [
                torch.ones(
                    t.shape[0],
                    (
                        self.n_registers + 1
                        if confidence is None
                        else self.n_registers + 2
                    ),
                    device=text_tokens.device,
                    dtype=torch.bool,
                ),
                text_tokens_mask,
            ],
            dim=1,
        )
        mask = mask.unsqueeze(1) * torch.ones(
            x.shape[0], x.shape[1], 1, device=text_tokens.device, dtype=torch.bool
        )
        for l in range(self.n_layers):

            x = self.layers[l](x, c, local_text_tokens, mask).clamp(-255.0, 255.0)
        # out modulation
        s, b = self.out_mod(c).chunk(2, dim=-1)
        out = modulation(self.out_ln(x), s, b)
        # remove registers
        out = out[:, self.n_registers :, :]
        # output proj
        out = self.out_proj(out)

        # depatchify
        out = einops.rearrange(
            out,
            "b (h w) (k s c) -> b c (h k) (w s)",
            h=self.n_patches,
            k=self.kernel_size,
            s=self.kernel_size,
        )

        return out


# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
