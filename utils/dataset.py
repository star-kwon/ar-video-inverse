from torch.utils.data import Dataset
import torch
from PIL import Image
import os
from torchvision.io import VideoReader

class VideoDataset(Dataset):
    def __init__(
        self,
        video_dir,
        num_frames=81,
        transform=None,
        extensions=(".mp4",),
        sort=True,
    ):
        assert os.path.isdir(video_dir), f"{video_dir} is not a directory"

        self.video_paths = [
            os.path.join(video_dir, fname)
            for fname in os.listdir(video_dir)
            if fname.lower().endswith(extensions)
        ]

        if sort:
            self.video_paths = sorted(self.video_paths)

        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        file_name = os.path.basename(video_path)
        prompt = os.path.splitext(file_name)[0]

        vr = VideoReader(video_path, "video")

        frames = []
        for i, frame in enumerate(vr):
            if i >= self.num_frames:
                break
            img = Image.fromarray(frame["data"].permute(1, 2, 0).numpy())
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

        video_tensor = torch.stack(frames, dim=0)

        return {
            "video": video_tensor,
            "prompts": prompt,
            "idx": idx,
        }
