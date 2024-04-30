import os
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm


def readscpfile(filename):
    audiofiles = open(filename, 'r').readlines()
    audiodict = []
    for file in audiofiles:
        noisyloc, cleanloc = file.strip().split(' ')
        audiodict.append({'noisy': noisyloc, 'clean': cleanloc})
    return audiodict


def getaudio(audioinfo, chunk_size):
    noisy = sf.read(audioinfo['noisy'])[0]
    clean = sf.read(audioinfo['clean'])[0]
    nsamples, csamples = noisy.shape[-1], clean.shape[-1]
    if nsamples >= csamples: noisy = noisy[:csamples]
    if csamples > nsamples:  clean = clean[:nsamples]
    samples = len(clean)
    if samples < chunk_size:
        r = chunk_size // samples + 1
        noisy = np.concatenate([noisy] * r)
        clean = np.concatenate([clean] * r)
    audio = np.vstack([noisy, clean])
    return audio


def makechunks(audio, chunk_size, hopsize):
    chunks = []
    nchannels, nsamples = audio.shape
    if nsamples < chunk_size:
        P = chunk_size - nsamples
        pad_width = ((0, 0), (0, P)) if nchannels >= 2 else (0, P)
        chunk = np.pad(audio, pad_width, "constant", constant_values=1.0e-8)
        ref = chunk[-1, :]
        # Consider a chunk only if the silence regions in the clean signal is less than 10% of the chunksize 
        if len(ref[abs(ref - 0.0) < 1.0e-6]) / (len(ref) + 1.0e-6) < 0.75:
            chunks.append(chunk)
    else:
        s = 0
        while True:
            if s + chunk_size > nsamples:
                break
            chunk = audio[:, s:s + chunk_size]
            ref = chunk[-1, :]
            # Consider a chunk only if the silence regions in the clean signal is less than 10% of the chunksize 
            if len(ref[abs(ref - 0.0) < 1.0e-6]) / (len(ref) + 1.0e-6) < 0.75:
                chunks.append(chunk)
            s += hopsize
    return chunks


def savechunks(scpfile, destdir, chunk_size=64000, hopsize=8000):
    Path(destdir).mkdir(parents=True, exist_ok=True)
    audiofiles = readscpfile(scpfile)
    chunk_count = 0
    for audiopair in tqdm(audiofiles, desc='Splitting audio '):
        audio = getaudio(audiopair, chunk_size)
        chunks = makechunks(audio, chunk_size, hopsize)
        for chunk in chunks:
            sf.write(destdir + '/audio_' + str(chunk_count) + '.wav', chunk.transpose(), 16000)
            chunk_count = chunk_count + 1
    return


if __name__ == '__main__':
    print('Datapreperation for Splitting audiofiles into 4 sec chunks')

    Path(os.path.dirname(os.getcwd() + '/Database/Training_Samples/Train')).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(os.getcwd() + '/Database/Training_Samples/Dev')).mkdir(parents=True, exist_ok=True)


    savechunks(os.getcwd() + '/scpfiles/Train.txt', os.getcwd() + '/Database/Training_Samples/Train')
    savechunks(os.getcwd() + '/scpfiles/Dev.txt', os.getcwd() + '/Database/Training_Samples/Dev')

