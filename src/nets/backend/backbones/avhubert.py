import torch
import numpy as np
from torch import nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import PreTrainedModel, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder, Wav2Vec2EncoderLayer,
    is_deepspeed_zero3_enabled
)
from copy import deepcopy
from transformers.modeling_outputs import BaseModelOutput
from src.nets.backend.backbones.resnet import ResEncoder
from src.nets.backend.backbones.modules.flash_attention_utils import index_first_axis, pad_input, _get_unpad_data
from src.nets.backend.backbones.modules.flash_attention_utils import FlashAttentionKwargs
from src.nets.backend.backbones.modules.avhubert_attention import AVHuBERTFlashAttention2


def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def compute_mask_indices(
        shape: Tuple[int, int],
        padding_mask: Optional[torch.Tensor],
        mask_prob: float,
        mask_length: int,
        mask_type: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 0,
        no_overlap: bool = False,
        min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    batch_indexes, starts, ends = [], [], []
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
        vals, run_starts, run_lengths = find_runs(mask[i])
        start_indices, lengths = run_starts[vals == True], run_lengths[vals == True]
        starts.append(start_indices)
        ends.append(start_indices + lengths)
        batch_indexes.append(np.zeros([len(start_indices)]) + i)
    return mask, np.concatenate(starts).astype(np.int64), np.concatenate(ends).astype(np.int64), np.concatenate(
        batch_indexes).astype(np.int64)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, cfg=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x


class AVHubertModel(PreTrainedModel):
    config_class = Wav2Vec2Config
    base_model_prefix = "avhubert"
    # _supports_flash_attn_2 = True

    # main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
            self,
            cfg: Wav2Vec2Config,
    ) -> None:
        super().__init__(cfg)
        # logger.info(f"HubertModel Config: {cfg}")

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / cfg.sample_rate
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers
        resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        self.feature_extractor_audio = SubModel(resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg)
        self.feature_extractor_video = SubModel(resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg)
        self.modality_dropout, self.audio_dropout = cfg.modality_dropout, cfg.audio_dropout
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == 'concat':
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == 'add':
            self.embed = cfg.encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob_image, self.mask_prob_audio = cfg.mask_prob_image, cfg.mask_prob_audio
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = cfg.mask_length_image, cfg.mask_length_audio
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type
        self.modality = cfg.modality

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_() if self.masking_type == 'input' else torch.FloatTensor(
                cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = AVHubertEncoder(cfg)

        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        # if self.untie_final_proj:
        #     self.final_proj = nn.Linear(
        #         cfg.encoder_embed_dim, final_dim * cfg.num_dictionaries
        #     )
        # else:
        #     self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        self.num_classes = [cfg.num_classes]
        self.label_embs_concat = nn.Parameter(
            torch.FloatTensor(sum(self.num_classes), final_dim)
        )
        nn.init.uniform_(self.label_embs_concat)

        # self._use_flash_attention_2 = False

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def apply_input_mask(self, x, padding_mask, target_list):
        B, C, T = x.shape[:3]
        is_audio = True if len(x.shape) == 3 else False
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        if mask_prob > 0:

            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices_np = mask_indices
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.transpose(1, 2).contiguous()  # [B, T, C, H, W]
            if B == 1:
                x[mask_indices] = 0
            elif is_audio:
                x[mask_indices] = self.mask_emb
            elif self.selection_type == 'same_other_seq':
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]
            elif self.selection_type == 'same_seq':
                batch_indexes_, other_indexes = [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    length = end - start
                    other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start - length), end))
                    if len(other_start) > 0:
                        other_start = np.random.choice(other_start, size=1)
                    else:
                        other_start = 0
                    other_end = other_start + length
                    other_indexes.append(np.arange(other_start, other_end).clip(max=T - 1))
                    batch_indexes_.append(np.zeros([length], dtype=np.int64) + batch_index)
                batch_indexes, other_indexes = np.concatenate(batch_indexes_), np.concatenate(other_indexes)
                x[mask_indices] = x[batch_indexes, other_indexes]

            x = x.transpose(1, 2).contiguous()
        else:
            mask_indices = None

        # if self.mask_channel_prob > 0:
        #     logger.info(f"No mask channel prob for input masking")
        return x, mask_indices

    def apply_feature_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        assert self.mask_prob_audio == self.mask_prob_image and self.mask_length_audio == self.mask_length_image, f"masking prob/length for image/audio be same for feature masking"
        mask_prob, mask_length = self.mask_prob_audio, self.mask_length_image
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_targets(
            self, features: torch.Tensor, mask_indices: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            if mask_indices is not None:
                mask_indices = mask_indices[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, mask_indices, target_list

    def forward_padding_mask(
            self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == 'dot':
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == 'cosine':
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1)  # [B*T, V]
            denom = (feats_ ** 2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (emb_mat ** 2).sum(dim=-1).sqrt().unsqueeze(
                dim=0)  # [B*T, V]
            logits = (nom / denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits

    def forward_gen(
            self,
            source: torch.Tensor,
            target_list: Optional[List[torch.Tensor]] = None,
            padding_mask: Optional[torch.Tensor] = None,
            mask: bool = True,
            features_only: bool = False,
            output_layer: Optional[int] = None,
            video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        features_audio = self.forward_features(src_audio, modality='audio')  # features: [B, F, T]
        features_video = self.forward_features(src_video, modality='video')

        if self.modality == 'audio':
            features_video = 0 * features_video
        elif self.modality == 'video':
            features_audio = 0 * features_audio
        else:
            if self.training:
                modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
                if modality_drop_prob < self.modality_dropout:
                    if audio_drop_prob < self.audio_dropout:
                        features_audio = 0 * features_audio
                    else:
                        features_video = 0 * features_video

        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        if target_list is not None:
            features, mask_indices, target_list = self.forward_targets(features, mask_indices, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if self.masking_type == 'feature' and mask:
            x, mask_indices = self.apply_feature_mask(features, padding_mask, target_list)
        else:
            x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        x = self.encoder(x, attention_mask=padding_mask)[0]
        # x = self.encoder(
        #     x,
        #     # attention_mask=padding_mask,
        #     # layer=None if output_layer is None else output_layer - 1
        # )[0]

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        proj_x = self.final_proj(x)
        if self.untie_final_proj:
            proj_x_list = proj_x.chunk(len(self.num_classes), dim=-1)
        else:
            proj_x_list = [proj_x for _ in self.num_classes]
        logit_list = [self.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in
                      zip(proj_x_list, label_embs_list, self.num_classes)]  # [[B*T, V]]
        mask, unmask = torch.logical_and(mask_indices, ~padding_mask).view(-1), torch.logical_and(~mask_indices,
                                                                                                  ~padding_mask).view(
            -1)  # [B*T]
        logit_m_list, logit_u_list = [logit[mask] for logit in logit_list], [logit[unmask] for logit in logit_list]
        target_m_list, target_u_list = [target.view(-1)[mask].long() for target in target_list], [
            target.view(-1)[unmask].long() for target in target_list]
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def forward(
            self,
            input_features: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            video: torch.Tensor = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward_gen(
            {"audio": input_features, "video": video},
            padding_mask=attention_mask,
            mask=False,
            features_only=True,
            output_layer=None,
        )
        feature = res["x"]
        return BaseModelOutput(last_hidden_state=feature, hidden_states=None, attentions=None)

    def extract_features(
            self,
            source: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            mask: bool = False,
            ret_conv: bool = False,
            output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward_gen(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def extract_finetune(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list=None)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list=None)
            mask_indices = torch.logical_or(mask_indices_audio,
                                            mask_indices_video)  # mask_indices not used in fine-tuning
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio')  # features: [B, F, T]
            features_video = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim,
                                                      features_audio.size(-1))
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = features_video.new_zeros(features_video.size(0), self.encoder_embed_dim,
                                                      features_video.size(-1))
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = self.forward_features(src_audio, modality='audio')  # features: [B, F, T]

        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        x = features
        mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x = self.encoder(
            x,
            # padding_mask=padding_mask,
            # layer=None if output_layer is None else output_layer - 1
        )[0]

        return x, padding_mask

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None

    def get_logits(self, net_output, is_masked=True):
        raise NotImplementedError

    def get_targets(self, net_output, is_masked=True):
        raise NotImplementedError

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits


class AVHubertEncoder(Wav2Vec2Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([AVHubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self._use_flash_attention_2 = (config._attn_implementation == "flash_attention_2")

    def forward(
            self,
            hidden_states: torch.tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        batch_size, seq_len, hidden_size = hidden_states.shape

        assert (self._attn_implementation == "flash_attention_2"), "only FlashAttention2 is currently supported."

        if self._attn_implementation == "flash_attention_2" and not self.gradient_checkpointing:

            # TODO: prepare data for flash attention
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=hidden_states.device)
            indices_k, cu_seq_lens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

            query_len = seq_len
            # else:

            packed_hidden_states = index_first_axis(hidden_states.view(-1, hidden_states.size(-1)), indices_k)
            flash_kwargs: FlashAttentionKwargs = {
                "cu_seq_lens_q": cu_seq_lens_k,
                "cu_seq_lens_k": cu_seq_lens_k,
                "max_length_q": max_seqlen_in_batch_k,
                "max_length_k": max_seqlen_in_batch_k,
                "query_length": query_len,
                "totals": packed_hidden_states.size(0)
            }

            hidden_states = packed_hidden_states
        else:
            indices_k = None
            flash_kwargs = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions,
                        flash_kwargs=flash_kwargs
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self._attn_implementation == "flash_attention_2" and not self.gradient_checkpointing:
            # pad the data again
            hidden_states = pad_input(hidden_states, indices_k, batch_size, seq_len)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class AVHubertEncoderLayer(Wav2Vec2EncoderLayer):

    def __init__(self, config):
        # config._attn_implementation = "eager"  # for debugging
        super().__init__(config)
        self._attn_implementation = config._attn_implementation

        # Override the attention module with your AVHubertFlashAttention2
        if config._attn_implementation == "flash_attention_2":
            self.attention = AVHuBERTFlashAttention2(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=False,
            )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False,
                flash_kwargs=None, **kwargs):

        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)

        assert self._attn_implementation == "flash_attention_2" or flash_kwargs is None
        attn_args = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
        }

        if self._attn_implementation == "flash_attention_2":
            attn_args["flash_kwargs"] = flash_kwargs

        hidden_states, attn_weights, _ = self.attention(
            **attn_args
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # hidden_states = self.layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)
        # hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
