from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .multihead_attention import MultiheadAttention
from .transformer_m_layers import AtomFeature, MoleculeAttnBias, Molecule3DBias, AtomTaskHead
from .transformer_m_encoder_layer import TransformerMEncoderLayer

def init_params(module):

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class TransformerMEncoder(nn.Module):

    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        encoder_normalize_before: bool = False,
        apply_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        add_3d: bool = False,
        num_3d_bias_kernel: int = 128,
        no_2d: bool = False,
        mode_prob: str = "0.2,0.2,0.6",
    ) -> None:

        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_init = apply_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.mode_prob = mode_prob

        try:
            mode_prob = [float(item) for item in mode_prob.split(',')]
            assert len(mode_prob) == 3
            assert sum(mode_prob) == 1.0
        except:
            mode_prob = [0.2, 0.2, 0.6]
        self.mode_prob = mode_prob

        self.atom_feature = AtomFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
        )

        self.molecule_attn_bias = MoleculeAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            no_2d=no_2d,
        )


        self.molecule_3d_bias = Molecule3DBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            n_layers=num_encoder_layers,
            embed_dim=embedding_dim,
            num_kernel=num_3d_bias_kernel,
            no_share_rpe=False,
        ) if add_3d else None

        self.atom_proc = AtomTaskHead(embedding_dim, num_attention_heads)

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        droppath_probs = [
            x.item() for x in torch.linspace(0, droppath_prob, num_encoder_layers)
        ]

        self.layers.extend(
            [
                self.build_transformer_m_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    sandwich_ln=sandwich_ln,
                    droppath_prob=droppath_probs[_],
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_init:
            self.apply(init_params)

    def build_transformer_m_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        sandwich_ln,
        droppath_prob,
    ):
        return TransformerMEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            sandwich_ln=sandwich_ln,
            droppath_prob=droppath_prob,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_tpu = False
        # compute padding mask. This is needed for multi-head attention

        data_x = batched_data["x"]
        n_mol, n_atom = data_x.size()[:2]
        padding_mask = (data_x[:,:,0]).eq(0) # B x T x 1
        padding_mask_cls = torch.zeros(n_mol, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B x (T+1) x 1
        mask_dict = {0: [1, 1], 1: [1, 0], 2: [0, 1]}
        mask_2d = mask_3d = None
        if self.training:
            mask_choice = np.random.choice(np.arange(3), n_mol, p=self.mode_prob)
            mask = torch.tensor([mask_dict[i] for i in mask_choice]).to(batched_data['pos'])
            mask_2d = mask[:, 0]
            mask_3d = mask[:, 1]

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.atom_feature(batched_data, mask_2d=mask_2d)

        if perturb is not None:
            x[:, 1:, :] += perturb

        # x: B x T x C

        attn_bias = self.molecule_attn_bias(batched_data, mask_2d=mask_2d)

        delta_pos = None
        if self.molecule_3d_bias is not None and not (batched_data["pos"] == 0).all():
            attn_bias_3d, merged_edge_features, delta_pos = self.molecule_3d_bias(batched_data)
            if mask_3d is not None:
                merged_edge_features, delta_pos = merged_edge_features * mask_3d[:, None, None], delta_pos * mask_3d[:, None, None, None]
                attn_bias_3d = attn_bias_3d.masked_fill_(((attn_bias_3d != float('-inf')) * (1 - mask_3d[:, None, None, None])).bool(), 0.0)
            attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] + attn_bias_3d
            x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)
            if not last_state_only:
                inner_states.append(x)

        atom_output = None
        if delta_pos is not None:
            atom_output = self.atom_proc(x[1:, :, :], attn_bias[:, :, 1:, 1:], delta_pos)
            if mask_3d is not None:
                mask_3d_only = (mask == torch.tensor([0.0, 1.0]).to(mask)[None, :]).all(dim=-1)
                atom_output = atom_output * mask_3d_only[:, None, None]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), atom_output
        else:
            return inner_states, atom_output
