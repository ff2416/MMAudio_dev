import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils
import os
from mmaudio.ext.mel_converter import get_mel_converter
from mmaudio.ext.autoencoder import AutoEncoderModule
import time
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import tqdm
import glob
log = logging.getLogger()

class Audio:
    def __init__(self, audio_path, sample_rate):
        self.audio_paths = audio_path
        self.sample_rate = sample_rate
        self.num_timbre_sample = 89088 if sample_rate == 44100 else 32768
        self.resampler = {}
    
    def load_audio(self):
        chunk_list=[]
        for audio_path in self.audio_paths:
            audio_chunk, sample_rate = torchaudio.load(audio_path)
            audio_chunk = audio_chunk.mean(dim=0)  # mono
            abs_max = audio_chunk.abs().max()
            audio_chunk = audio_chunk / abs_max * 0.95

            # resample
            if sample_rate == self.sample_rate:
                audio_chunk = audio_chunk
            else:
                if sample_rate not in self.resampler:
                    # https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
                    self.resampler[sample_rate] = torchaudio.transforms.Resample(
                        sample_rate,
                        self.sample_rate,
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        resampling_method='sinc_interp_kaiser',
                        beta=14.769656459379492,
                    )
                audio_chunk = self.resampler[sample_rate](audio_chunk)
            if audio_chunk.size(0) < self.num_timbre_sample:
                padding_length = self.num_timbre_sample - audio_chunk.size(0)
                audio_chunk = torch.cat([audio_chunk, torch.zeros(padding_length)], dim=0)
            else:
                audio_chunk = audio_chunk[:self.num_timbre_sample]
            # audio_chunk = audio_chunk[:self.num_timbre_sample]
            chunk_list.append(audio_chunk)
        return chunk_list
    
def process_video(video_path: Path, args, model: ModelConfig, net: MMAudio, fm: FlowMatching, feature_utils: FeaturesUtils, device: str, dtype: torch.dtype, audio: torch.Tensor, i):
    log.info(f'Processing video: {video_path}')
    t=time.time()
    audio_num_sample = 89088
    if audio is not None:
        audio_num_sample = audio.shape[0]
    video_info = load_video(video_path, args.duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    if args.mask_away_clip:
        clip_frames = None
    else:
        clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)

    model.seq_cfg.duration = duration
    model.seq_cfg.audio_num_sample = audio_num_sample
    net.update_seq_lengths(model.seq_cfg.latent_seq_len, model.seq_cfg.clip_seq_len, model.seq_cfg.sync_seq_len, model.seq_cfg.audio_seq_len)

    log.info(f'Prompt: {args.prompt}')
    log.info(f'Negative prompt: {args.negative_prompt}')
    audios = generate(clip_frames,
                      sync_frames, [args.prompt], audio,
                      negative_text=[args.negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=torch.Generator(device=device).manual_seed(args.seed),
                      cfg_strength=args.cfg_strength)
    audio = audios.float().cpu()[0]
    save_path = args.output / f'{video_path.stem}{i}.wav'
    torchaudio.save(save_path, audio, model.seq_cfg.sampling_rate)
    log.info(f'Audio saved to {save_path}')

    if not args.skip_video_composite:
        video_save_path = args.output / f'{video_path.stem}{i}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=model.seq_cfg.sampling_rate)
        log.info(f'Video saved to {video_save_path}')

@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='large_44k',
                        help='small_16k, small_44k, medium_44k, large_44k, large_44k_v2')
    parser.add_argument('--video_dir', type=Path, help='')
    parser.add_argument('--audio_path', type=str, default='')
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--mask_away_clip', action='store_true')
    parser.add_argument('--output', type=Path, help='Output directory', default='./cartoons/run')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--full_precision', action='store_true')

    args = parser.parse_args()

    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]
    model.download_if_needed()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    args.output.mkdir(parents=True, exist_ok=True)

    if  args.audio_path != '':
        SAMPLE_RATE = 44100
        audio = Audio([args.audio_path], SAMPLE_RATE)
        audio_list = audio.load_audio()

    else:
        audio_list = None
    model.model_path = Path('/project/llmsvgen/pengjun/MMAudio_dev/output/random2_finetune_global/random2_finetune_global_ckpt_280000.pth')
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True)['weights'])
    log.info(f'Loaded weights from {model.model_path}')

    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=args.num_steps)
    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=True)
    feature_utils = feature_utils.to(device, dtype).eval()

    if args.video_dir:
        video_dir: Path = args.video_dir.expanduser()
        video_files = sorted(list(video_dir.glob('*.mp4')))
        if os.path.isfile(args.video_dir):
            video_files=[args.video_dir]
        if not video_files:
            log.warning(f'No video files found in {video_dir}')
        else:
            if audio_list is None:
                audio_list = [None] * len(video_files)
            if len(audio_list)==1:
                audio_list = audio_list * len(video_files)
            for i in range(10):
                for video_path, audio in tqdm.tqdm(zip(video_files,audio_list)):
                    args.seed = torch.seed()
                    process_video(video_path, args, model, net, fm, feature_utils, device, dtype, audio, i)
    else:
        log.info('No video directory provided -- text-to-audio mode')


if __name__ == '__main__':
    main()