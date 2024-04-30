import os
from pathlib import Path
## -------------------------------------  Train -----------------------------------------------

SNR_all = [0, 5]
Noise_type_all = ['Babble', 'Car']
Audio_sample = 10
rootdir = os.getcwd() + '/Database/Original_Samples/'
Path(os.path.dirname(os.getcwd() +'/scpfiles/')).mkdir(parents=True, exist_ok=True)
file = open(os.getcwd() +'/scpfiles/Train.txt','w')

for Noise_type in Noise_type_all:
    for SNR in SNR_all:
        for i in range(Audio_sample):
            signal_path = rootdir + '/Train/Clean/Clean_' + str(i + 1) + '.wav'
            noisy_path = rootdir + '/Train/Noisy/Noisy_' + Noise_type + '_' + str(SNR) + '_dB_' + str(i + 1) + '.wav'
            L = noisy_path + " " + signal_path
            file.writelines(L)
            file.write('\n')
file.close()

# ##-------------------------------------  Dev ------------------------------------------------------

SNR_all = [0, 5]
Noise_type_all = ['Babble', 'Car']
Audio_sample = 5
file = open(os.getcwd() +'/scpfiles/Dev.txt','w')
for Noise_type in Noise_type_all:
    for SNR in SNR_all:
        for i in range(Audio_sample):
            signal_path = rootdir + '/Dev/Clean/Clean_' + str(i + 1) + '.wav'
            noisy_path = rootdir + '/Dev/Noisy/Noisy_' + Noise_type + '_' + str(SNR) + '_dB_' + str(i + 1) + '.wav'
            L = noisy_path + " " + signal_path
            file.writelines(L)
            file.write('\n')
file.close()

## -------------------------------------  Eval -----------------------------------------------------

SNR_all = [0, 5]
Noise_type_all  = ['Babble', 'Car']
Audio_sample = 5
file = open(os.getcwd() +'/scpfiles/Test.txt','w')
for Noise_type in Noise_type_all:
    for SNR in SNR_all:
        for i in range(Audio_sample):
            Enhanced_path = rootdir + '/Test/Enhanced/Enhanced_' + Noise_type + '_' + str(SNR) + '_dB_' + str(i+1) + '.wav'
            signal_path = rootdir + '/Test/Clean/Clean_' + str(i + 1) + '.wav'
            noisy_path = rootdir + '/Test/Noisy/Noisy_' + Noise_type + '_' + str(SNR) + '_dB_' + str(i+1) + '.wav'
            L = noisy_path + " " + signal_path + " " + Enhanced_path
            file.writelines(L)
            file.write('\n')
file.close()
