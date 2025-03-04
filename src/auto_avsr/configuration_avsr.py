from src.nets.backend.e2e_asr_conformer_av import E2E
from transformers.configuration_utils import PretrainedConfig

class AutoAVSRConfig(PretrainedConfig):
    model_type = "auto_avsr"

    def __init__(
        self,
        odim=5049,
        adim=768,
        aheads=12,
        eunits=3072,
        elayers=12,
        transformer_input_layer="conv3d",
        dropout_rate=0.1,
        transformer_attn_dropout_rate=0.1,
        transformer_encoder_attn_layer_type="rel_mha",
        macaron_style=True,
        use_cnn_module=True,
        cnn_module_kernel=31,
        zero_triu=False,
        a_upsample_ratio=1,
        relu_type="swish",
        ddim=768,
        dheads=12,
        dunits=3072,
        dlayers=6,
        lsm_weight=0.1,
        transformer_length_normalized_loss=False,
        mtlalpha=0.1,
        ctc_type="builtin",
        rel_pos_type="latest",
        aux_adim=768,
        aux_aheads=12,
        aux_eunits=3072,
        aux_elayers=12,
        aux_transformer_input_layer="conv1d",
        aux_dropout_rate=0.1,
        aux_transformer_attn_dropout_rate=0.1,
        aux_transformer_encoder_attn_layer_type="rel_mha",
        aux_macaron_style=True,
        aux_use_cnn_module=True,
        aux_cnn_module_kernel=31,
        aux_zero_triu=False,
        aux_a_upsample_ratio=1,
        aux_relu_type="swish",
        aux_dunits=3072,
        aux_dlayers=6,
        aux_lsm_weight=0.1,
        aux_transformer_length_normalized_loss=False,
        aux_mtlalpha=0.1,
        aux_ctc_type="builtin",
        aux_rel_pos_type="latest",
        fusion_hdim=8192,
        fusion_norm="batchnorm",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.odim = odim
        self.adim = adim
        self.aheads = aheads
        self.eunits = eunits
        self.elayers = elayers
        self.transformer_input_layer = transformer_input_layer
        self.dropout_rate = dropout_rate
        self.transformer_attn_dropout_rate = transformer_attn_dropout_rate
        self.transformer_encoder_attn_layer_type = transformer_encoder_attn_layer_type
        self.macaron_style = macaron_style
        self.use_cnn_module = use_cnn_module
        self.cnn_module_kernel = cnn_module_kernel
        self.zero_triu = zero_triu
        self.a_upsample_ratio = a_upsample_ratio
        self.relu_type = relu_type
        self.ddim = ddim
        self.dheads = dheads
        self.dunits = dunits
        self.dlayers = dlayers
        self.lsm_weight = lsm_weight
        self.transformer_length_normalized_loss = transformer_length_normalized_loss
        self.mtlalpha = mtlalpha
        self.ctc_type = ctc_type
        self.rel_pos_type = rel_pos_type
        self.aux_adim = aux_adim
        self.aux_aheads = aux_aheads
        self.aux_eunits = aux_eunits
        self.aux_elayers = aux_elayers
        self.aux_transformer_input_layer = aux_transformer_input_layer
        self.aux_dropout_rate = aux_dropout_rate
        self.aux_transformer_attn_dropout_rate = aux_transformer_attn_dropout_rate
        self.aux_transformer_encoder_attn_layer_type = aux_transformer_encoder_attn_layer_type
        self.aux_macaron_style = aux_macaron_style
        self.aux_use_cnn_module = aux_use_cnn_module
        self.aux_cnn_module_kernel = aux_cnn_module_kernel
        self.aux_zero_triu = aux_zero_triu
        self.aux_a_upsample_ratio = aux_a_upsample_ratio
        self.aux_relu_type = aux_relu_type
        self.aux_dunits = aux_dunits
        self.aux_dlayers = aux_dlayers
        self.aux_lsm_weight = aux_lsm_weight
        self.aux_transformer_length_normalized_loss = aux_transformer_length_normalized_loss
        self.aux_mtlalpha = aux_mtlalpha
        self.aux_ctc_type = aux_ctc_type
        self.aux_rel_pos_type = aux_rel_pos_type
        self.fusion_hdim = fusion_hdim
        self.fusion_norm = fusion_norm
        
        