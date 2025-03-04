import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from src.dataset.avhubert_dataset import load_audio, load_video, cut_or_pad, AudioTransform, VideoTransform, DataCollator
from src.tokenizer.spm_tokenizer import TextTransform
from src.avhubert_avsr.avhubert_avsr_model import AVHubertAVSR, get_beam_search_decoder
from src.avhubert_avsr.configuration_avhubert_avsr import AVHubertAVSRConfig
from datasets import load_from_disk
import torch

def inference(model, video, audio, text_transform, beam_search):
    avhubert_features = model.encoder(
        input_features = audio, 
        video = video,
    )
    audiovisual_feat = avhubert_features.last_hidden_state

    audiovisual_feat = audiovisual_feat.squeeze(0)

    nbest_hyps = beam_search(audiovisual_feat)
    nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
    predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
    predicted = text_transform.post_process(predicted_token_id).replace("<eos>", "")
    return predicted

if __name__ == "__main__":
    # Load text transform
    sp_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram_vn/unigram2048.model")
    dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/tokenizer/spm/unigram_vn/unigram2048_units.txt")
    text_transform = TextTransform(
        sp_model_path=sp_model_path,
        dict_path=dict_path,
    )
    
    # Load data collator
    audio_transform = AudioTransform(subset="test")
    video_transform = VideoTransform(subset="test")
    
    av_data_collator = DataCollator(
        text_transform=text_transform,
        audio_transform=audio_transform,
        video_transform=video_transform,
    )
    
    # Load model
    model_name = "nguyenvulebinh/AV-HuBERT-CTC-Attention-VI"
    avsr_model = AVHubertAVSR.from_pretrained(model_name, cache_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model-bin"))
    avsr_model.eval().cuda()
    beam_search = get_beam_search_decoder(avsr_model.avsr, text_transform.token_list)
    
    # Load sample
    raw_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/dummy")
    metadata_path = os.path.join(raw_data_path, "metadata")
    dataset = load_from_disk(metadata_path)['test'].map(lambda x: {"video": os.path.join(raw_data_path, x["video"])})
    
    # Inference
    sample = dataset[0]
    sample_features = av_data_collator([sample])
    audios = sample_features["audios"].cuda()
    videos = sample_features["videos"].cuda()
    audio_lengths = sample_features["audio_lengths"].cuda()
    video_lengths = sample_features["video_lengths"].cuda()
    
    output = inference(avsr_model.avsr, videos, audios, text_transform, beam_search)
    
    print("ref: " + sample['label'])
    print("hyp: " + output)