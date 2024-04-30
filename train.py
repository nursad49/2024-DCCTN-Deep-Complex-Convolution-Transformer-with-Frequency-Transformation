import os, argparse, shutil
import torch, math
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import loaddataset
from auraloss.time import SISDRLoss
from auraloss.freq import STFTLoss
import warnings
warnings.filterwarnings("ignore")
epsilon = torch.finfo(torch.float32).eps


class DeepLearningModel(pl.LightningModule):
    def __init__(self, net, batch_size=1):
        super(DeepLearningModel, self).__init__()
        self.model = net
        self.modelname = self.model.name
        self.batch_size = batch_size
        self.si_sdr = SISDRLoss()
        self.freqloss = STFTLoss(fft_size=320, hop_size=80, win_length=320, sample_rate=16000, scale_invariance=False,
                                 w_sc=0.0)
        print('\nUsing Si-SDR + STFT loss function to train the network! ....')

    def forward(self, x):
        return self.model(x)

    def loss_function(self, cln_audio, enh_audio):
        loss = self.si_sdr(cln_audio, enh_audio) + 25 * self.freqloss(cln_audio, enh_audio)
        return loss

    def training_step(self, batch, batch_nb):
        enh_audio = self(batch['noisy'])
        loss = self.loss_function(batch['clean'], enh_audio)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        enh_audio = self(batch['noisy'])
        loss = self.loss_function(batch['clean'], enh_audio)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-5, betas=(0.5, 0.999))
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True),
                     'interval': 'epoch', 'frequency': 1, 'reduce_on_plateau': True, 'monitor': 'val_loss'}
        return [optimizer], [scheduler]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Enhancement usnig SkipConv Net')
    parser.add_argument('--model', type=str, help='ModelName', default='DCCTN_Modified')
    parser.add_argument('--mode', type=str, help='Choose between "summary", "fast_run" or "train"', default='summary')
    parser.add_argument('--b', type=int, help='Batch Size', default=8)
    parser.add_argument('--e', type=int, help='Epocs', default=2)
    parser.add_argument('--gpu', type=str, help='GPU-IDs used to Train', default='0')
    parser.add_argument('--loss', type=str, help='loss function', default='SISDR+FreqLoss')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'CFTNet':
        from Network import CFTNet
        curr_model = CFTNet()
    elif args.model == 'DCCTN':
        from Network import DCCTN
        curr_model = DCCTN
    elif args.model == 'DATCFTNET':
        from Network import DATCFTNET
        curr_model = DATCFTNET()
    elif args.model == 'DATCFTNET_DSC':
        from Network import DATCFTNET_DSC
        curr_model = DATCFTNET_DSC()

    print("This process has the PID", os.getpid())
    model = DeepLearningModel(curr_model, args.b)
    print('Training Model: ' + model.modelname)
    gpuIDs = [int(k) for k in args.gpu.split(' ')]
    print('Training on GPU(s) : ', gpuIDs)
    print('save model to : ' + os.getcwd() + '/Saved_Models/' + model.modelname)
    callbacks = ModelCheckpoint(monitor='val_loss', dirpath=os.getcwd() + '/Saved_Models/' + model.modelname,
                                filename=model.modelname + '-DADX-IEEE-' + args.loss + '-{epoch:02d}-{val_loss:.2f}',
                                save_top_k=1, mode='min')
    TrainData = loaddataset(os.getcwd() + '/Database/Training_Samples/Train')
    trainloader = DataLoader(TrainData, batch_size=args.b, shuffle=True, num_workers=12, pin_memory=True)
    DevData = loaddataset(os.getcwd() + '/Database/Training_Samples/Dev')
    devloader = DataLoader(DevData, batch_size=args.b, shuffle=False, num_workers=12, pin_memory=True)
    print(gpuIDs)
    print(torch.cuda.is_available())
    trainer = pl.Trainer(max_epochs=args.e, gpus=gpuIDs, strategy='ddp', callbacks=callbacks, gradient_clip_val=10,
                         accumulate_grad_batches=8)  # GPU only
    # trainer = pl.Trainer(max_epochs=args.e, accelerator="cpu", callbacks=callbacks, gradient_clip_val=10, accumulate_grad_batches=8) # CPU only
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=devloader)
    # shutil.rmtree(os.getcwd()+'/lightning_logs')
    print('Done!')

# ------------------  How to Train -----------------------------------------------------
# Run the following code in the terminal
# python3 train.py --model "$model name" --b "$batch size" --e "$Num of epoch" --loss "$Loss Function" --gpu "$GPUs"
# Example: python3 train.py --model CFTNet --b 8 --e 50 --loss SISDR+FreqLoss --gpu '0 1'
# Simple Example: python3 train.py --model CFTNet
