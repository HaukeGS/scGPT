from functools import lru_cache
import math
from typing import Dict, Optional
import json

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones

from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attention import FlashAttention
from flash_attn.modules.mha import FlashCrossAttention
from .layers import MultiheadAttention
from .moe import MoE

class FlashscGPTMHA(nn.Module):
    """
    Custom MHA layer for scGPT. This takes two separate forward passes on the pect
    genes, and on the gen genes.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
    ) -> None:
        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.self_attn = FlashAttention(attention_dropout=attention_dropout)
        self.cross_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )
        # self.cross_attn = FlashCrossAttention(attention_dropout=attention_dropout)
        # for cross attetion, launch multiple queries in parallel, each query is just
        # a single gen gene. Then each kv is the entire set of pect genes plus this gen
        # gene together.
        # In practice, we can simply put these queries in the batch dimension, and then
        # they can be processed in parallel.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        need_weights=False,
    ):
        """
        pcpt_total_embs: (batch, pcpt_len, hidden_dim) (where hidden_dim = num heads * head dim)
        gen_total_embs: (batch, gen_len, hidden_dim)
        pcpt_key_padding_mask: bool tensor of shape (batch, pcpt_len), 1 means flat_mask and 0 means not flat_mask.
        gen_key_padding_mask: bool tensor of shape (batch, gen_len), 1 means flat_mask and 0 means not flat_mask.
        """
        pcpt_qkv = self.Wqkv(pcpt_total_embs)

        pcpt_qkv = rearrange(
            pcpt_qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )

        # full self attention on pcpt genes
        pcpt_context, pcpt_attn_weights = self.self_attn(
            pcpt_qkv,
            key_padding_mask=pcpt_key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        pcpt_context = self.out_proj(rearrange(pcpt_context, "b s h d -> b s (h d)"))

        if gen_total_embs is None:
            return (pcpt_context, None), (pcpt_attn_weights, None)

        gen_qkv = self.Wqkv(gen_total_embs)
        gen_qkv = rearrange(
            gen_qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )

        # CROSS ATTENTION USING RAW PYTORCH IMPLEMENTATION
        cross_q = gen_qkv[:, :, 0, :, :]  # (batch, gen_len, nheads, head_dim)
        cross_q = rearrange(cross_q, "b gen_s h d -> b gen_s (h d)")
        cross_kv = torch.cat(
            [pcpt_qkv[:, :, 1:, :, :], gen_qkv[:, :, 1:, :, :]], dim=1
        )  # (batch, pcpt_seq+gen_seq, 2, nheads, head_dim)
        cross_kv = rearrange(cross_kv, "b pcpt_gen_s two h d -> b pcpt_gen_s two (h d)")

        # make the attention mask, for pytorch implementation, true means attention is not allowed
        @lru_cache(maxsize=1)
        def make_mask(q_len, k_len, device):
            attention_mask = torch.zeros(
                (q_len, k_len), device=device, dtype=torch.bool
            )  # (gen_len, pcpt_len+gen_len)
            # make the last gen_len by gen_gen to be true, only the diagonal is allowed with false
            attention_mask[:, -q_len:] = ~torch.eye(
                q_len, device=device, dtype=torch.bool
            )
            return attention_mask

        attention_mask = make_mask(cross_q.shape[1], cross_kv.shape[1], cross_q.device)

        if pcpt_key_padding_mask is None and gen_key_padding_mask is None:
            key_padding_mask = None
        else:
            if pcpt_key_padding_mask is None:
                pcpt_key_padding_mask = torch.ones(
                    (pcpt_qkv.shape[0], pcpt_qkv.shape[1]),
                    device=pcpt_qkv.device,
                    dtype=torch.bool,
                )
            elif gen_key_padding_mask is None:
                gen_key_padding_mask = torch.ones(
                    (gen_qkv.shape[0], gen_qkv.shape[1]),
                    device=gen_qkv.device,
                    dtype=torch.bool,
                )
            key_padding_mask = ~torch.cat(
                [pcpt_key_padding_mask, gen_key_padding_mask], dim=1
            )
        cross_context, _ = self.cross_attn(
            cross_q,
            cross_kv[:, :, 0, :],
            cross_kv[:, :, 1, :],
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
        )
        gen_context = cross_context  # (batch, gen_len, hidden_dim)
        gen_attn_weights = None

        # # CROSS ATTENTION ON GEN GENES
        # # prepare cross_q, where each query is per only one gen gene
        # cross_q = gen_qkv[:, :, 0, :, :]  # (batch, gen_len, nheads, head_dim)
        # cross_q = rearrange(cross_q, "b s h d -> b s (h d)")
        # cross_q_unpad, indices_q, cu_seq_len_q, max_seqlen_q = unpad_input(
        #     cross_q, gen_key_padding_mask
        # )
        # # if not care about padding (b gen_s) 1 (h d)

        # # the input to cross attention, q needs to be (total_q, nheads, head_dim)

        # # prepare gen_kv, where each kv is this gen gene plus the entire set of pect genes
        # # if not care about padding (b gen_s) pcpt_seq+1 (h d)

        # # call cross attention
        # gen_context, gen_attn_weights = self.cross_attn(
        #     gen_q,
        #     gen_kv,
        #     q_padding_mask=gen_key_padding_mask,
        #     kv_padding_mask=pcpt_key_padding_mask,
        #     need_weights=need_weights,
        # )
        # # rearrange output to (batch, gen_len, hidden_dim)

        # # TEMP TEST
        # gen_context, gen_attn_weights = self.self_attn(
        #     gen_qkv,
        #     key_padding_mask=gen_key_padding_mask,
        #     need_weights=need_weights,
        #     causal=self.causal,
        # )
        # gen_context = self.out_proj(rearrange(gen_context, "b s h d -> b s (h d)"))

        return (pcpt_context, gen_context), (pcpt_attn_weights, gen_attn_weights)


class FlashscGPTLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
        num_experts=0,
        k=0
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = FlashscGPTMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        if num_experts > 0 and k <= 0 or num_experts <= 0 and k > 0:
            raise ValueError("Both num_experts and k should be provided together")
        if num_experts > 0 and k > 0:
            self.moe = MoE(input_size=d_model, output_size=d_model, num_experts=num_experts, hidden_size=dim_feedforward, k=k, noisy_gating=True)
            self.moe = self.moe.to(device)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def _reverse_key_padding_mask(self, src_key_padding_mask):
        """
        Reverse the true false values of the key padding mask. This is because
        we follow pytorch rule that the mask is True for padded tokens, but
        in the inner flash MHA, it assumes the mask is False for padded tokens.
        """
        if src_key_padding_mask is None:
            return None

        if not src_key_padding_mask.any().item():
            # no padding tokens in src
            return None
        return ~src_key_padding_mask
    

    def _apply_moe_masked(
        self,
        embs: Tensor,                          # (batch_size, seq_len, embed_dim)
        key_padding_mask: Optional[Tensor],   # (batch_size, seq_len), True = padded
        gene_ids: Optional[Tensor] = None,          # (batch_size, seq_len)
        cell_type_ids: Optional[Tensor] = None,    # (batch_size, 1)
        expert_specialization: Optional[bool] = False
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, embed_dim = embs.shape
        flat_embeddings = embs.reshape(batch_size * seq_len, embed_dim) # (batch_size * seq_len, embed_dim)

        if key_padding_mask is None:
            raise ValueError("key_padding_mask cannot be None for _apply_moe_masked, since we need to know which rows are valid for MoE")
            # out_flat, aux_loss, rows_indices_per_expert = self.moe(flat_embeddings, True if expert_specialization_params is not None else False)
            # return out_flat.reshape(batch_size, seq_len, embed_dim), aux_loss, rows_indices_per_expert if expert_specialization_params else None # TODO

        flat_mask = (~key_padding_mask).reshape(batch_size * seq_len)  # True = real token
        out_flat = torch.zeros_like(flat_embeddings)
        flat_embeddings_masked = flat_embeddings[flat_mask]

        if flat_mask.any():
            out_valid, aux_loss, rows_indices_per_expert = self.moe(flat_embeddings_masked, expert_specialization)
            out_flat[flat_mask] = out_valid
        else:
            aux_loss = None

        return out_flat.reshape(batch_size, seq_len, embed_dim), aux_loss, rows_indices_per_expert


    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        cell_type_ids: Optional[Tensor] = None,
        pcpt_genes: Optional[Tensor] = None,
        gen_genes: Optional[Tensor] = None,
        expert_specialization: Optional[bool] = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        pcpt_key_padding_mask_ = self._reverse_key_padding_mask(pcpt_key_padding_mask)
        gen_key_padding_mask_ = self._reverse_key_padding_mask(gen_key_padding_mask)

        if self.norm_scheme == "pre":
            print(f"stepping in norm_scheme: pre")
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            if gen_total_embs is not None:
                gen_total_embs = self.norm1(gen_total_embs)
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)
            pcpt_total_embs2 = self.linear2(
                self.dropout(self.activation(self.linear1(pcpt_total_embs)))
            )
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)
                gen_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(gen_total_embs)))
                )
                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
        else:
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
            )[0]


            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm1(pcpt_total_embs)

            if hasattr(self, "moe") and self.moe is not None:
                # Mixture of Experts
                pcpt_total_embs2, aux_loss_pcpt, row_indices_per_expert_pcpt = self._apply_moe_masked(pcpt_total_embs, pcpt_key_padding_mask, pcpt_genes, cell_type_ids, expert_specialization)
            else:
                pcpt_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(pcpt_total_embs)))
                )
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)

            aux_loss_gen = None
            if gen_total_embs is not None:

                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm1(gen_total_embs)

                if hasattr(self, "moe") and self.moe is not None:
                    # Mixture of Experts
                    gen_total_embs2, aux_loss_gen, row_indices_per_expert_gen = self._apply_moe_masked(gen_total_embs, gen_key_padding_mask, gen_genes, cell_type_ids, expert_specialization)
                else:
                    gen_total_embs2 = self.linear2(
                        self.dropout(self.activation(self.linear1(gen_total_embs)))
                    )

                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)

        if hasattr(self, "moe") and self.moe is not None:
            if aux_loss_gen is not None:
                aux_loss = aux_loss_pcpt + aux_loss_gen
            else:
                aux_loss = aux_loss_pcpt
        else:
            aux_loss = None
            row_indices_per_expert_pcpt = None
            row_indices_per_expert_gen = None
        return pcpt_total_embs, gen_total_embs, aux_loss, row_indices_per_expert_pcpt, row_indices_per_expert_gen


class FlashscGPTGenerator(nn.Module):
    # takes in the set of different inputs in an mapping
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        mask_check=True,
        # expert_specialization_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check
        self.gene_label_layer_distributions = None
        self.cell_type_label_layer_distributions = None


    def init_label_distributions(self, expert_specialization_params):
        n_experts = expert_specialization_params.get("n_experts", None)
        n_genes = expert_specialization_params.get("n_genes", None)
        n_cell_types = expert_specialization_params.get("n_cell_types", None)
        if n_experts is None or n_genes is None or n_cell_types is None:
            print(f"Could not initialize label distributions because of missing parameters in expert_specialization_params: {expert_specialization_params}")
            return
        if self.gene_label_layer_distributions is None:
            self.gene_label_layer_distributions = torch.zeros(
                len(self.layers), 
                n_experts, 
                n_genes,
                dtype=torch.int64,
                device=next(self.layers.parameters()).device
            )
        if self.cell_type_label_layer_distributions is None:
            self.cell_type_label_layer_distributions = torch.zeros(
                len(self.layers), 
                n_experts,
                n_cell_types,
                dtype=torch.int64,
                device=next(self.layers.parameters()).device
            )


    def save_label_distribution(
        self, 
        expert_specialization_params,
        row_indices_per_expert_pcpt,
        row_indices_per_expert_gen,
        layer,
        pcpt_total_embs,
        pcpt_key_padding_mask,
        gen_total_embs,
        gen_key_padding_mask,
        pcpt_genes,
        gen_genes,
        cell_type_ids,
    ):
        if self.gene_label_layer_distributions is None or self.cell_type_label_layer_distributions is None:
            self.init_label_distributions(expert_specialization_params)
        batch_size_pcpt, seq_len_pcpt, _ = pcpt_total_embs.shape
        batch_size_gen, seq_len_gen, _ = gen_total_embs.shape
        flat_mask_pcpt = (~pcpt_key_padding_mask).reshape(batch_size_pcpt * seq_len_pcpt)  # True = real token
        flat_mask_gen = (~gen_key_padding_mask).reshape(batch_size_gen * seq_len_gen)  # True = real token
        gene_ids_flat_pcpt = pcpt_genes.reshape(batch_size_pcpt * seq_len_pcpt).squeeze() if pcpt_genes is not None else None
        gene_ids_flat_masked_pcpt = gene_ids_flat_pcpt[flat_mask_pcpt]
        gene_ids_flat_gen = gen_genes.reshape(batch_size_gen * seq_len_gen).squeeze() if gen_genes is not None else None
        gene_ids_flat_masked_gen = gene_ids_flat_gen[flat_mask_gen]
        cell_type_ids_flat_pcpt = cell_type_ids.repeat_interleave(seq_len_pcpt, dim=0).squeeze() if cell_type_ids is not None else None
        cell_type_ids_flat_masked_pcpt = cell_type_ids_flat_pcpt[flat_mask_pcpt]
        cell_type_ids_flat_gen = cell_type_ids.repeat_interleave(seq_len_gen, dim=0).squeeze() if cell_type_ids is not None else None
        cell_type_ids_flat_masked_gen = cell_type_ids_flat_gen[flat_mask_gen]
        n_experts = expert_specialization_params.get("n_experts", None)
        k = expert_specialization_params.get("k", None)
        n_genes = expert_specialization_params.get("n_genes", None)
        n_cell_types = expert_specialization_params.get("n_cell_types", None)
        # debugging asserts
        # assert(flat_mask_pcpt.shape == gene_ids_flat_pcpt.shape == cell_type_ids_flat_pcpt.shape)
        # assert(flat_mask_gen.shape == gene_ids_flat_gen.shape == cell_type_ids_flat_gen.shape)
        # assert(gene_ids_flat_masked_pcpt.shape == cell_type_ids_flat_masked_pcpt.shape)
        # assert(n_experts is not None and k is not None and n_genes is not None and n_cell_types is not None)
        # assert(len(row_indices_per_expert_pcpt) == n_experts)
        # assert(len(row_indices_per_expert_gen) == n_experts)

        # assert(sum(expert.shape[0] for expert in row_indices_per_expert_pcpt) == gene_ids_flat_masked_pcpt.shape[0] * k)
        # assert(max(rows.max().item() for rows in [rows for rows in row_indices_per_expert_pcpt if rows.numel() > 0]) == gene_ids_flat_masked_pcpt.shape[0]-1)
        # assert(sum(expert.shape[0] for expert in row_indices_per_expert_gen) == gene_ids_flat_masked_gen.shape[0] * k)
        # assert(max(rows.max().item() for rows in [rows for rows in row_indices_per_expert_gen if rows.numel() > 0]) == gene_ids_flat_masked_gen.shape[0]-1)

        if row_indices_per_expert_pcpt is not None:
            for expert_id, rows in enumerate(row_indices_per_expert_pcpt):
                if rows.numel() > 0:
                    gene_labels = gene_ids_flat_masked_pcpt[rows]
                    self.gene_label_layer_distributions[layer, expert_id] += torch.bincount(gene_labels, minlength=n_genes)
                    cell_type_labels = cell_type_ids_flat_masked_pcpt[rows]
                    self.cell_type_label_layer_distributions[layer, expert_id] += torch.bincount(cell_type_labels, minlength=n_cell_types)
        if row_indices_per_expert_gen is not None:
            for expert_id, rows in enumerate(row_indices_per_expert_gen):
                if rows.numel() > 0:
                    gene_labels = gene_ids_flat_masked_gen[rows]
                    self.gene_label_layer_distributions[layer, expert_id] += torch.bincount(gene_labels, minlength=n_genes)
                    cell_type_labels = cell_type_ids_flat_masked_gen[rows]
                    self.cell_type_label_layer_distributions[layer, expert_id] += torch.bincount(cell_type_labels, minlength=n_cell_types)


    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        cell_type_ids: Optional[Tensor] = None,
        pcpt_genes: Optional[Tensor] = None,
        gen_genes: Optional[Tensor] = None,
        expert_specialization_params: Optional[Dict] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if pcpt_key_padding_mask is not None:
            _skpm_dtype = pcpt_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                pcpt_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        running_aux_loss = None
        for i, mod in enumerate(self.layers):
            # pcpt_total_embs, gen_total_embs, aux_loss, gene_label_distribution, cell_type_label_distribution = mod(
            pcpt_total_embs, gen_total_embs, aux_loss, row_indices_per_expert_pcpt, row_indices_per_expert_gen = mod(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask,
                gen_key_padding_mask,
                cell_type_ids,
                pcpt_genes,
                gen_genes,
                True if expert_specialization_params is not None else False
            )
            running_aux_loss = aux_loss if running_aux_loss is None else running_aux_loss + aux_loss
            if expert_specialization_params is not None:
                self.save_label_distribution(
                    expert_specialization_params,
                    row_indices_per_expert_pcpt,
                    row_indices_per_expert_gen,
                    i,
                    pcpt_total_embs,
                    pcpt_key_padding_mask,
                    gen_total_embs,
                    gen_key_padding_mask,
                    pcpt_genes,
                    gen_genes,
                    cell_type_ids,
                )

        if self.norm is not None:
            pcpt_total_embs = self.norm(pcpt_total_embs)
            gen_total_embs = self.norm(gen_total_embs)

        return pcpt_total_embs, gen_total_embs, running_aux_loss, self.gene_label_layer_distributions, self.cell_type_label_layer_distributions