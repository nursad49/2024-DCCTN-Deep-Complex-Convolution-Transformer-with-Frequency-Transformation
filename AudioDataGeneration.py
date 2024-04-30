import librosa as lr
import numpy as np
import soundfile as sf
import os
from pathlib import Path


def rms_energy(x):
    return 10 * np.log10((1e-12 + x.dot(x)) / len(x))


def SPL_cal(x, SPL):
    SPL_before = 20 * np.log10(np.sqrt(np.mean(x ** 2)) / (20 * 1e-6))
    y = x * 10 ** ((SPL - SPL_before) / 20)
    return y


def add_noise(signal, noise, fs, snr, signal_energy='rms'):
    # Generate random section of masker
    if len(noise) != len(signal):
        idx = np.random.randint(0, len(noise) - len(signal))
        noise = noise[idx:idx + len(signal)]
    # Scale noise wrt speech at target SNR
    N_dB = rms_energy(noise)
    if signal_energy == 'rms':
        S_dB = rms_energy(signal)
    else:
        raise ValueError('signal_energy has to be either "rms" or "P.56"')

    # Rescale N
    N_new = S_dB - snr
    noise_scaled = 10 ** (N_new / 20) * noise / 10 ** (N_dB / 20)
    noisy_signal = signal + noise_scaled
    return noisy_signal


def datagenerator(In_path, Out_path, Noise_file, SNR, Num_audio_sample, sample_margin, Fs):
    for index in range(Num_audio_sample):
        # ** data for input **
        Out_file_noisy = "Noisy_" + Noise_file + "_" + str(SNR) + "_dB_" + str(index + sample_margin)
        Out_file_clean = "Clean_" + str(index + sample_margin)
        clean, Fs = lr.load(In_path + '/Clean/Clean_' + str(index + sample_margin) + '.wav',
                            sr=Fs)  # Downscale 48kHz to 16kHz
        noise, Fs = lr.load(In_path + '/Different_Noise/' + Noise_file + '_noise.wav',
                            sr=Fs)  # Downscale 48kHz to 16kHz

        # x = y + Noise #Add noise
        clean = SPL_cal(clean, 65)
        noisy = add_noise(clean, noise, Fs, SNR, signal_energy='rms')
        noisy = SPL_cal(noisy, 65)
        sf.write(Out_path + '/Noisy/' + Out_file_noisy + ".wav", noisy, Fs)
        sf.write(Out_path + '/Clean/' + Out_file_clean + ".wav", clean, Fs)
        print(str(index))


if __name__ == '__main__':
    path = os.getcwd()
    Data_path = path + '/Database/Original_Samples'
    # Out_path = path + '/Database/Original_Samples'
    Path(os.path.dirname(Data_path + '/Train/Clean/')).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(Data_path + '/Train/Noisy/')).mkdir(parents=True, exist_ok=True)

    Path(os.path.dirname(Data_path + '/Dev/Clean/')).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(Data_path + '/Dev/Noisy/')).mkdir(parents=True, exist_ok=True)

    Path(os.path.dirname(Data_path + '/Test/Clean/')).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(Data_path + '/Test/Noisy/')).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(Data_path + '/Test/Enhanced/')).mkdir(parents=True, exist_ok=True)
    # --------------------------------------  Train -------------------------------------------------------------------
    Fs = 16000
    print("Started ....................")
    Noise_type = ['Babble', 'Car']
    SNR_all = [0, 5]  
    Num_audio_sample = 10
    sample_margin=1
    for Noise_file in Noise_type:
        for SNR in SNR_all:
            datagenerator(Data_path, Data_path + '/Train/', Noise_file, SNR, Num_audio_sample,sample_margin, Fs)
    print("Completed generating Train data ................... ")


    ##--------------------------------------  Dev -------------------------------------------------------------------
    Noise_type = ['Babble', 'Car']
    SNR_all = [0, 5]
    Num_audio_sample = 5
    sample_margin = 11 # As number of train sample was 10
    for Noise_file in Noise_type:
        for SNR in SNR_all:
            datagenerator(Data_path, Data_path + '/Dev/', Noise_file, SNR, Num_audio_sample,sample_margin, Fs)
    print("Completed generating Dev data ................... ")

    ##--------------------------------------  Test --------------------------------------------------------------------
    Noise_type = ['Babble', 'Car']
    Num_audio_sample = 5
    sample_margin = 16 # As number of train+ Dev sample was 15
    SNR_all = [5, 10]
    for Noise_file in Noise_type:
        for SNR in SNR_all:
            datagenerator(Data_path, Data_path + '/Test/', Noise_file, SNR, Num_audio_sample, sample_margin, Fs)
    print("Completed generating Test data ................... ")
