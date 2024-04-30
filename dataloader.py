import os
import numpy as np
import soundfile as sf
from glob import glob
from librosa import stft
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.image as mpimg
from pathlib import Path
import warnings


warnings.filterwarnings("ignore")
epsilon = np.finfo(np.float32).eps


class loaddataset(Dataset):
    def __init__(self, datadir=os.getcwd() + '/Training_Samples/Train'):
        self.audiofiles = sorted(glob(datadir + '/*.wav'))

    def __len__(self): return len(self.audiofiles)

    def __getitem__(self, idx):
        print(self.audiofiles[idx]); audio = sf.read(self.audiofiles[idx])[0]
        noisy, clean = audio[:, 0].transpose(), audio[:, 1]
        noisy, clean = torch.FloatTensor(noisy).squeeze(), torch.FloatTensor(clean).squeeze()
        batch = {'noisy': noisy, 'clean': clean}
        return batch


def norm(x): return x / (np.max(np.abs(x)) + 1e-10)


def lps(audio):
    spec = norm(stft(audio, n_fft=512, hop_length=128, win_length=512, window=np.sqrt(np.hanning(512))))
    spec = db(spec)
    return spec


def db(x):
    xdb = 20 * np.log10(np.abs(x) + np.spacing(1))
    xdb[xdb < -120] = -120
    xdb = (xdb + 60) / 60
    return xdb


if __name__ == '__main__':
    # Test Dataloader
    Path(os.getcwd() + '/output').mkdir(parents=True, exist_ok=True)
    dataset = loaddataset(os.getcwd() + '/Training_Samples/Train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
    for i, batch in enumerate(dataloader):
        if i == 10:
            print(batch['noisy'].shape, batch['clean'].shape)
            nspec = lps(batch['noisy'][0].cpu().numpy())
            cspec = lps(batch['clean'][0].cpu().numpy())
            mpimg.imsave(os.getcwd() + '/output/noisy_mag.png', np.flipud(nspec), cmap='jet')
            mpimg.imsave(os.getcwd() + '/output/clean_mag.png', np.flipud(cspec), cmap='jet')
            break
