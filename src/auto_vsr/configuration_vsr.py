from transformers.configuration_utils import PretrainedConfig

class AutoVSRConfig(PretrainedConfig):
    model_type = "auto_vsr"

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
        
        