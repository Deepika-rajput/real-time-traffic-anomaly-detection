import os
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for label, folder in enumerate(['accident','normal']):
            path = os.path.join(root, folder)
            for clip in os.listdir(path):
                self.samples.append((os.path.join(path, clip), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_path, label = self.samples[idx]
        frames = sorted(os.listdir(clip_path))

        imgs = []
        for f in frames:
            img = cv2.imread(os.path.join(clip_path, f))
            img = cv2.resize(img, (224,224))
            img = img / 255.0
            imgs.append(img)

        imgs = torch.tensor(imgs).permute(0,3,1,2).float()
        return imgs, torch.tensor(label)