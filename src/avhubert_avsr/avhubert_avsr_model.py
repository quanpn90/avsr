from src.nets.backend.e2e_asr_avhubert import E2E
from transformers.modeling_utils import PreTrainedModel
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
from src.nets.batch_beam_search import BatchBeamSearch
from src.nets.scorers.length_bonus import LengthBonus
from src.nets.scorers.ctc import CTCPrefixScorer
import torch
from transformers.utils import ModelOutput
from typing import List, Optional, Union
from dataclasses import dataclass

def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=3):
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

@dataclass
class AVHubertAVSROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_ctc: Optional[torch.FloatTensor] = None
    loss_att: Optional[torch.FloatTensor] = None
    acc: Optional[torch.FloatTensor] = None

class AVHubertAVSR(PreTrainedModel):
    config_class = AVHubertAVSRConfig
    
    def __init__(self, config: AVHubertAVSRConfig):
        super().__init__(config)
        self.avsr = E2E(config)
    
    def forward(self, 
        videos, 
        audios, 
        labels,
        video_lengths, 
        audio_lengths, 
        label_lengths
    ):
        loss, loss_ctc, loss_att, acc = self.avsr(videos, audios, video_lengths, audio_lengths, labels)
        return AVHubertAVSROutput(
            loss=loss,
            loss_ctc=loss_ctc,
            loss_att=loss_att,
            acc=acc
        )
        # return self.avsr(videos, audios, video_lengths, audio_lengths, labels)
    
    
    
    # def inference(self, video, audio, text_transform):
    #     self.beam_search = get_beam_search_decoder(self.avsr, text_transform.token_list)
    #     video_feat, _ = self.avsr.encoder(video.unsqueeze(0).to(self.device), None)
    #     audio_feat, _ = self.avsr.aux_encoder(audio.unsqueeze(0).to(self.device), None)
    #     audiovisual_feat = self.avsr.fusion(torch.cat((video_feat, audio_feat), dim=-1))

    #     audiovisual_feat = audiovisual_feat.squeeze(0)

    #     nbest_hyps = self.beam_search(audiovisual_feat)
    #     nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
    #     predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
    #     predicted = text_transform.post_process(predicted_token_id).replace("<eos>", "")
    #     return predicted