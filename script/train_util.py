
from safetensors.torch import load_file
import json, os
from peft import LoraModel, LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

def load_sharded_state_dict(model_dir):
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    state_dict = {}
    for shard in set(index["weight_map"].values()):
        state_dict.update(load_file(os.path.join(model_dir, shard)))
    return state_dict


def create_lora(avsr_model):
    lora_target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(r=16, lora_alpha=64,
                             target_modules=lora_target_modules, lora_dropout=0.05,
                             bias="none")

    avsr_model.language_model.add_adapter(lora_config)
    avsr_model.audio_tower.add_adapter(lora_config)

    # avsr_model.language_model = get_peft_model(avsr_model.language_model, lora_config)
    # avsr_model.audio_tower = get_peft_model(avsr_model.audio_tower, lora_config)

    return avsr_model


def merge_lora(avsr_model):
    # avsr_model.language_model.merge_and_unload()
    # avsr_model.audio_tower.merge_and_unload()

    def recursively_merge_lora_layers(module):
        for _, submodule in module.named_modules():
            if isinstance(submodule, LoraLayer):
                if not submodule.merged:
                    submodule.merge()

    recursively_merge_lora_layers(avsr_model.language_model)
    recursively_merge_lora_layers(avsr_model.audio_tower)f

    return avsr_model

# state_dict = load_sharded_state_dict("path/to/checkpoint_with_lora")
# model.load_state_dict(state_dict, strict=False)