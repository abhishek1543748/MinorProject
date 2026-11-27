"""
ssl_aasist.py — wav2vec 2.0 XLS-R (300M) + AASIST back-end ("SSL-AASIST")
==========================================================================
Port of TakHemlata/SSL_Anti-spoofing model.py (Tak et al., Odyssey 2022,
https://arxiv.org/abs/2202.12233) WITHOUT fairseq (no Py3.13/Windows build).

  - Back-end (GraphAttentionLayer … Model): same computation as upstream
    model.py, comments/debug prints removed. Upstream quirks are KEPT because
    the pretrained weights were trained with them (e.g. Residual_block.conv1
    reads x, not the bn1 output; HtrgGAT layers get self.master1/2 directly).
    Differences: Model.__init__ args, forward() returns (last_hidden, output)
    to match src/model/AASIST.py, SSL front-end kept in eval mode.
  - Front-end: SSLModel on transformers.Wav2Vec2Model with the XLS-R 300M
    architecture inlined (no download at inference). Upstream checkpoints
    store the fine-tuned fairseq wav2vec2 under "ssl_model.model.*";
    load_ssl_aasist() renames them to HF names and loads with strict=True,
    so any mapping gap fails loudly.

Class convention is the same as AASIST: output index 0 = spoof, 1 = bonafide.
Input: [B, 64600] float32 at 16 kHz (same preprocess.py as AASIST-L).
"""

import os
import re
from typing import Union

os.environ.setdefault("USE_TF", "0")   # transformers: skip TensorFlow import

import torch
import torch.nn as nn
import torch.nn.functional as F


# XLS-R 300M architecture (facebook/wav2vec2-xls-r-300m config.json).
# Dropout/LayerDrop/masking are irrelevant in eval mode but set to 0 anyway.
XLSR_300M = dict(
    conv_bias=True, conv_dim=[512] * 7, conv_kernel=[10, 3, 3, 3, 3, 2, 2],
    conv_stride=[5, 2, 2, 2, 2, 2, 2], do_stable_layer_norm=True,
    feat_extract_activation="gelu", feat_extract_norm="layer", hidden_act="gelu",
    hidden_size=1024, intermediate_size=4096, layer_norm_eps=1e-5,
    num_attention_heads=16, num_conv_pos_embedding_groups=16,
    num_conv_pos_embeddings=128, num_feat_extract_layers=7, num_hidden_layers=24,
    apply_spec_augment=False, layerdrop=0.0, hidden_dropout=0.0,
    attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0,
)


class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        self.model = Wav2Vec2Model(Wav2Vec2Config(**XLSR_300M))
        self.out_dim = 1024

    def extract_feat(self, input_data):
        # (batch, length) -> (batch, frames, 1024); fairseq features_only=True ['x']
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        return self.model(input_data).last_hidden_state


# ── AASIST back-end (upstream) ──────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        return self.act(x)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = self.bn(x.view(-1, org_size[-1]))
        return x.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.)

    def forward(self, x1, x2, master=None):
        num_type1, num_type2 = x1.size(1), x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        return x * x.transpose(1, 2)

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)
        att_map = att_board / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = self.bn(x.view(-1, org_size[-1]))
        return x.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        scores = self.sigmoid(self.proj(self.drop(h)))
        return self.top_k_graph(scores, h, self.k)

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3),
                               padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3),
                               padding=(0, 1), stride=1)
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], padding=(0, 1),
                                             kernel_size=(1, 3), stride=1)

    def forward(self, x):
        identity = x
        # Upstream quirk kept for weight compatibility: bn1/selu output is
        # computed but conv1 reads x directly.
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.ssl_model = SSLModel()
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x = self.LL(x_ssl_feat)                 # (bs, frames, 128)

        x = x.transpose(1, 2).unsqueeze(dim=1)  # (bs, 1, 128, frames)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # spectral attention
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # temporal attention
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # inference 1 (upstream passes self.master1, not the expanded copy)
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        return last_hidden, output


# ── Checkpoint loading (fairseq names → transformers names) ─────────────────

_RENAMES = [
    (r"^post_extract_proj\.", "feature_projection.projection."),
    (r"^layer_norm\.", "feature_projection.layer_norm."),
    (r"^mask_emb$", "masked_spec_embed"),
    (r"^feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv."),
    (r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.", r"feature_extractor.conv_layers.\1.layer_norm."),
    (r"^encoder\.pos_conv\.0\.", "encoder.pos_conv_embed.conv."),
    (r"^encoder\.layers\.(\d+)\.self_attn\.", r"encoder.layers.\1.attention."),
    (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"encoder.layers.\1.layer_norm."),
    (r"^encoder\.layers\.(\d+)\.fc1\.", r"encoder.layers.\1.feed_forward.intermediate_dense."),
    (r"^encoder\.layers\.(\d+)\.fc2\.", r"encoder.layers.\1.feed_forward.output_dense."),
    (r"^encoder\.layers\.(\d+)\.final_layer_norm\.", r"encoder.layers.\1.final_layer_norm."),
    (r"^encoder\.layer_norm\.", "encoder.layer_norm."),
]
# fairseq pre-training heads, unused for features_only=True
_DROP = re.compile(r"^(quantizer|project_q|final_proj|label_embs_concat)")


def _convert_ssl_key(k: str, hf_keys: set) -> str | None:
    for pat, rep in _RENAMES:
        new, n = re.subn(pat, rep, k)
        if n:
            k = new
            break
    else:
        return None if _DROP.match(k) else k
    # weight-norm pos_conv: fairseq weight_g/weight_v; newer torch keeps
    # parametrizations.weight.original0/original1 instead
    if k.endswith(("weight_g", "weight_v")) and k not in hf_keys:
        k = (k.replace("weight_g", "parametrizations.weight.original0")
              .replace("weight_v", "parametrizations.weight.original1"))
    return k


def quantize_ssl(model: nn.Module) -> nn.Module:
    """
    Dynamic INT8 quantization of the wav2vec2 front-end's nn.Linear layers
    (attention q/k/v/out + feed-forward: ~99% of the 317M params), in place.

    Only model.ssl_model is quantized. The AASIST back-end stays fp32 because
    dynamically quantized Linear has no autograd, and Grad-CAM needs
    gradients from the logits back to model.encoder's output (that path is
    back-end only). The conv feature extractor / pos_conv stay fp32 too
    (quantize_dynamic does not cover Conv).

    CPU only, inference only. Scores move slightly vs fp32, so a calibrator
    fitted on fp32 scores is approximate for the quantized model — refit
    with scripts/refit_windowed_calibrator.py --model ssl --quantize.
    """
    from torch.ao.quantization import quantize_dynamic

    engines = torch.backends.quantized.supported_engines
    for eng in ("x86", "fbgemm", "qnnpack"):
        if eng in engines:
            torch.backends.quantized.engine = eng
            break
    model.eval()
    # inplace=True: the default deep-copies the 1.2 GB fp32 front-end first
    quantize_dynamic(model.ssl_model, {nn.Linear}, dtype=torch.qint8, inplace=True)
    # Front-end is frozen: without this, its fp32 LayerNorm params make autograd
    # trace (and try to backprop) through the INT8 ops, which have no autograd
    # kernel. Grad-CAM only needs gradients in the back-end.
    for p in model.ssl_model.parameters():
        p.requires_grad_(False)
    model.quantized = True
    return model


def load_ssl_aasist(weights_path, device="cpu", quantize: bool = False) -> Model:
    """Build SSL-AASIST and load an upstream checkpoint (strict).
    quantize=True applies quantize_ssl() (CPU INT8 front-end)."""
    # Low-RAM load: build the graph on the meta device (no 1.2 GB random init)
    # and assign the checkpoint tensors directly as parameters (no copy).
    # Each checkpoint tensor has its own storage, so the in-place quantize
    # frees fp32 Linear weights layer by layer. (Not mmap=True: all mmap'd
    # tensors share ONE file mapping that stays alive while any fp32 param does.)
    from transformers import Wav2Vec2Model  # noqa: F401  (import must not run under meta)
    with torch.device("meta"):
        model = Model()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}

    hf_keys = set(model.ssl_model.model.state_dict().keys())
    out = {}
    for k, v in state.items():
        if k.startswith("ssl_model.model."):
            nk = _convert_ssl_key(k[len("ssl_model.model."):], hf_keys)
            if nk is not None:
                out["ssl_model.model." + nk] = v
        else:
            out[k] = v

    model.load_state_dict(out, strict=True, assign=True)
    del state, out
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if t.is_meta]
    if leftover:
        raise RuntimeError(f"tensors not loaded from checkpoint: {leftover[:5]}")
    if quantize:
        if torch.device(device).type != "cpu":
            raise ValueError("quantize=True is CPU-only (dynamic INT8)")
        return quantize_ssl(model)
    return model.to(device).eval()
