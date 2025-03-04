from src.nets.backend.e2e_asr_conformer import E2E
from transformers.modeling_utils import PreTrainedModel
from src.auto_asr.configuration_asr import AutoASRConfig
from src.nets.batch_beam_search import BatchBeamSearch
from src.nets.scorers.length_bonus import LengthBonus
from src.nets.scorers.ctc import CTCPrefixScorer
import torch

def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=10):
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )


class AutoASR(PreTrainedModel):
    config_class = AutoASRConfig
    
    def __init__(self, config: AutoASRConfig):
        super().__init__(config)
        self.asr = E2E(config)
    
    def forward(self, 
        video, audio, video_lengths, audio_lengths, label
    ):
        return self.asr(video, audio, video_lengths, audio_lengths, label)
    
    def inference(self, sample, text_transform):
        
        self.beam_search = get_beam_search_decoder(self.asr, text_transform.token_list)
        enc_feat, _ = self.asr.encoder(sample.unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted