import json
import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
import torchaudio
from mmaudio.utils.dist_utils import local_rank
import random
log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class VideoDataset(Dataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        *,
        duration_sec: float = 8.0,
    ):
        self.video_root = Path(video_root)

        self.duration_sec = duration_sec
        self.sample_rate = 44100
        self.resampler = {}
        self.expected_audio_length = 89088
        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.clip_transform = v2.Compose([
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # to be implemented by subclasses
        self.captions = {}
        self.videos = sorted(list(self.captions.keys()))

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        caption = self.captions[video_id]

        reader = StreamingMediaDecoder(self.video_root / (video_id + '.mp4'))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )
        reader.add_basic_audio_stream(frames_per_chunk=2**30, )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]
        audio_chunk = data_chunk[2]
        
        if clip_chunk is None:
            raise RuntimeError(f'CLIP video returned None {video_id}')
        if clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(
                f'CLIP video too short {video_id}, expected {self.clip_expected_length}, got {clip_chunk.shape[0]}'
            )

        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_id}')
        if sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f'Sync video too short {video_id}, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}'
            )

         # process audio
        sample_rate = int(reader.get_out_stream_info(2).sample_rate)
        audio_chunk = audio_chunk.transpose(0, 1)
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

        if audio_chunk.shape[0] < self.expected_audio_length:
            raise RuntimeError(f'Audio too short {video_id}')
        # start_index = random.randint(0, audio_chunk.shape[0] - self.expected_audio_length)
        timbre_sample = audio_chunk[audio_chunk.shape[0]-self.expected_audio_length:] 

        # truncate the video
        clip_chunk = clip_chunk[:self.clip_expected_length]
        if clip_chunk.shape[0] != self.clip_expected_length:
            raise RuntimeError(f'CLIP video wrong length {video_id}, '
                               f'expected {self.clip_expected_length}, '
                               f'got {clip_chunk.shape[0]}')
        clip_chunk = self.clip_transform(clip_chunk)

        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(f'Sync video wrong length {video_id}, '
                               f'expected {self.sync_expected_length}, '
                               f'got {sync_chunk.shape[0]}')
        sync_chunk = self.sync_transform(sync_chunk)

        data = {
            'name': video_id,
            'caption': caption,
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
            'audio': timbre_sample
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.videos[idx]}: {e}')
            return None

    def __len__(self):
        return len(self.captions)


class VGGSound(VideoDataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        csv_path: Union[str, Path],
        *,
        duration_sec: float = 8.0,
    ):
        super().__init__(video_root, duration_sec=duration_sec)
        self.video_root = Path(video_root)
        self.csv_path = Path(csv_path)

        videos = sorted(os.listdir(self.video_root))
        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')
        self.captions = {}

        df = pd.read_csv(csv_path, header=None, names=['id', 'sec', 'caption',
                                                       'split']).to_dict(orient='records')

        videos_no_found = []
        for row in df:
            if row['split'] == 'test':
                start_sec = int(row['sec'])
                video_id = str(row['id'])
                # this is how our videos are named
                video_name = f'{video_id}_{start_sec:06d}'
                if video_name + '.mp4' not in videos:
                    videos_no_found.append(video_name)
                    continue

                self.captions[video_name] = row['caption']

        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')
            log.info(f'{len(self.captions)} useable videos found')
            if videos_no_found:
                log.info(f'{len(videos_no_found)} found in {csv_path} but not in {video_root}')
                log.info(
                    'A small amount is expected, as not all videos are still available on YouTube')

        self.videos = sorted(list(self.captions.keys()))


class MovieGen(VideoDataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        jsonl_root: Union[str, Path],
        *,
        duration_sec: float = 10.0,
    ):
        super().__init__(video_root, duration_sec=duration_sec)
        self.video_root = Path(video_root)
        self.jsonl_root = Path(jsonl_root)

        videos = sorted(os.listdir(self.video_root))
        videos = [v[:-4] for v in videos]  # remove extensions
        self.captions = {}

        for v in videos:
            with open(self.jsonl_root / (v + '.jsonl')) as f:
                data = json.load(f)
                self.captions[v] = data['audio_prompt']

        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {video_root}')

        self.videos = videos
