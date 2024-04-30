import torch, os, sys
import soundfile as sf
from pathlib import Path
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.image as mpimg
import numpy as np
from librosa import stft, istft, griffinlim
from pystoi import stoi
from pesq import pesq
from auraloss.time import SISDRLoss, SNRLoss
from Objective_metrics import *
import speechmetrics
from torch import nn
import scipy
from Network import *


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred, actual))


class test_model:
    def __init__(self, modelname, modelfile, evalfile, Loss_function):
        super(test_model, self).__init__()
        self.evalfile = evalfile
        self.modelname = modelname
        self.modelfile = modelfile
        self.Loss_function = Loss_function
        # self.si_sdr = SISDRLoss()
        self.rmse = RMSE()
        self.SNRLoss = SNRLoss()

    def mapmodule(self, state_dict, keyword='model'):
        model_dict = []
        for key in state_dict.keys():
            if keyword in key:
                new_key = key.replace(keyword + '.', '')
                model_dict.append((new_key, state_dict[key]))
        new_state_dict = OrderedDict(model_dict)
        return new_state_dict

    def loadmodel(self):
        if self.modelname == 'CFTNet':
            from Network import CFTNet
            bestmodel = glob(os.getcwd() + '/Saved_Models/CFTNet/' + self.modelfile)[0]
            model = CFTNet()
        elif self.modelname == 'DCCTN':
            from Network import DCCTN
            bestmodel = glob(os.getcwd() + '/Saved_Models/DCCTN/' + self.modelfile)[0]
            model = DCCTN()
        elif self.modelname == 'DATCFTNET':
            from Network import DATCFTNET
            bestmodel = glob(os.getcwd() + '/Saved_Models/DATCFTNET/' + self.modelfile)[0]
            model = DATCFTNET()
        elif self.modelname == 'DATCFTNET_DSC':
            from Network import DATCFTNET_DSC
            bestmodel = glob(os.getcwd() + '/Saved_Models/DATCFTNET_DSC/' + self.modelfile)[0]
            model = DATCFTNET_DSC()

        print('Initializing model :  ' + self.modelname)
        print('Loading weights    :  ' + bestmodel)
        saved_model = torch.load(bestmodel, map_location='cpu')
        saved_model['state_dict'] = self.mapmodule(saved_model['state_dict'])
        model.load_state_dict(saved_model['state_dict'])
        model = model.to('cuda')
        model.eval()
        return model

    def main(self):
        model = self.loadmodel()
        eval_files = open(self.evalfile).readlines()
        eval_files = [i.strip().split(' ') for i in eval_files]
        STOI_C_N = np.zeros(len(eval_files))
        STOI_C_E = np.zeros(len(eval_files))
        PESQ_C_N = np.zeros(len(eval_files))
        PESQ_C_E = np.zeros(len(eval_files))
        LSD_C_N = np.zeros(len(eval_files))
        LSD_C_E = np.zeros(len(eval_files))
        SNRloss_C_N = np.zeros(len(eval_files));
        SNRloss_C_E = np.zeros(len(eval_files))
        SISDR_C_N = np.zeros(len(eval_files))
        SISDR_C_E = np.zeros(len(eval_files))
        RMSE_C_N = np.zeros(len(eval_files));
        RMSE_C_E = np.zeros(len(eval_files))

        Path(os.path.dirname(os.getcwd() + '/Result/')).mkdir(parents=True, exist_ok=True)
        j = 0
        for audiofile in tqdm(eval_files, desc='Decoidng Eval Files '):
            noisy = self.norm(sf.read(audiofile[0])[0])
            clean = self.norm(sf.read(audiofile[1])[0])
            clean = self.SPL_cal(clean, 65);
            noisy = self.SPL_cal(noisy, 65)
            # enh_audio = (model(torch.FloatTensor(noisy).unsqueeze(0).to('cuda')).detach().cpu().numpy()).squeeze(0)
            enh_audio = model(torch.FloatTensor(noisy).unsqueeze(0).to('cuda')).detach().cpu().numpy()
            # enh_audio=enh_audio.squeeze(0)
            enh_audio = self.norm(self.match_dims(noisy, enh_audio))
            enh_audio = self.SPL_cal(enh_audio, 65)
            desname = audiofile[2]
            sf.write(desname, enh_audio, 16000)
            Path(os.path.dirname(desname)).mkdir(parents=True, exist_ok=True)
            window_length = 5
            metrics = speechmetrics.load(['sisdr'], window_length)  # ['bsseval', 'sisdr', 'stoi', 'pesq']
            scores_C_N, scores_C_E = metrics(audiofile[0], audiofile[1]), metrics(audiofile[2], audiofile[
                1])  # (path_to_estimate_file, path_to_reference)
            SISDR_C_N[j], SISDR_C_E[j] = scores_C_N['sisdr'][0], scores_C_E['sisdr'][0]
            RMSE_C_N[j], RMSE_C_E[j] = self.rmse(torch.from_numpy(clean), torch.from_numpy(noisy)), self.rmse(
                torch.from_numpy(clean), torch.from_numpy(enh_audio))
            PESQ_C_N[j], PESQ_C_E[j] = pesq(16000, clean, noisy, 'wb'), pesq(16000, clean, enh_audio, 'wb')
            STOI_C_N[j], STOI_C_E[j] = stoi(clean, noisy, 16000, extended=False), stoi(clean, enh_audio, 16000,
                                                                                       extended=False)
            LSD_C_N[j], LSD_C_E[j] = lsd(clean, noisy), lsd(clean, enh_audio)
            SNRloss_C_N[j], SNRloss_C_E[j] = self.SNRLoss(torch.from_numpy(clean),
                                                          torch.from_numpy(noisy)), self.SNRLoss(
                torch.from_numpy(clean), torch.from_numpy(enh_audio))

            print('Sample: ' + str(j) + ' The Noisy PESQ score: ' + str(
                (PESQ_C_N[j])) + '\nThe Enhanced PESQ score: ' + str((PESQ_C_E[j])))
            print('The Noisy STOI score: ' + str((STOI_C_N[j])) + '\nThe Enhanced STOI score: ' + str((STOI_C_E[j])))
            print('The Noisy LSD score: ' + str((LSD_C_N[j])) + '\nThe Enhanced LSD score: ' + str((LSD_C_E[j])))
            j += 1
        print('The mean Noisy STOI score: ' + str(np.mean(STOI_C_N)) + '\nThe mean Enhanced STOI score: ' + str(
            np.mean(STOI_C_E)))
        print('The mean Noisy PESQ score: ' + str(np.mean(PESQ_C_N)) + '\nThe mean Enhanced PESQ score: ' + str(
            np.mean(PESQ_C_E)))
        print('The mean Noisy SISDR score: ' + str(np.mean(SISDR_C_N)) + '\nThe mean Enhanced SISDR score: ' + str(
            np.mean(SISDR_C_E)))
        print('The mean Noisy LSD score: ' + str(np.mean(LSD_C_N)) + '\nThe mean Enhanced LSD score: ' + str(
            np.mean(LSD_C_E)))
        print(
            'The mean Noisy SNRLoss score: ' + str(np.mean(SNRloss_C_N)) + '\nThe mean Enhanced SNRLoss score: ' + str(
                np.mean(SNRloss_C_E)))
        print('The mean Noisy RMSE score: ' + str(np.mean(RMSE_C_N)) + '\nThe mean Enhanced RMSE score: ' + str(
            np.mean(RMSE_C_E)))
        np.savez(os.getcwd() + '/Result/IEEE_Objective_score_' + self.modelname + '_' + self.Loss_function, PESQ_C_N,
                 PESQ_C_E, STOI_C_N, STOI_C_E, LSD_C_N, LSD_C_E, SISDR_C_N, SISDR_C_E, RMSE_C_N, RMSE_C_E, SNRloss_C_N,
                 SNRloss_C_E)

    def match_dims(self, rev, enh):
        output = np.zeros_like(rev)
        if len(enh) >= len(rev):
            output = enh[:len(rev)]
        if len(enh) < len(rev):
            output[:len(enh)] = enh
        return output

    def norm(self, x):
        return x / (np.max(np.abs(x)) + 1e-10)

    def SPL_cal(self, x, SPL):
        SPL_before = 20 * np.log10(np.sqrt(np.mean(x ** 2)) / (20 * 1e-6))
        y = x * 10 ** ((SPL - SPL_before) / 20)
        return y


if __name__ == '__main__':
    print('Testing the model .................')
    modelname = 'DCCTN'
    modelfile = 'DCCTN-DADX-IEEE-SISDR+FreqLoss-epoch=45-val_loss=-2.93.ckpt'
    Loss_function = 'SISDR+FreqLoss=-2.93'
    evalfile = os.getcwd() + '/scpfiles/Test.txt'
    test = test_model(modelname, modelfile, evalfile, Loss_function)
    test.main()
