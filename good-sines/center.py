import pytorch_lightning as pl
import torch
from waveflow import WaveFlow, TacotronSTFT, WaveFlowLossDataParallel
from audible_sines import AudibleSines

class GoodSines(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.stft = TacotronSTFT(
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            sampling_rate=44100,
            mel_fmin=0.0, 
            mel_fmax=8000.0)
        self.waveflow = WaveFlow()
        self.loss = WaveFlowLossDataParallel()

    def training_step(self, batch, batch_idx):
        melspec = self.stft.mel_spectrogram(batch)
        y_hat, logdet = self.waveflow(melspec)
        loss = self.loss(y_hat, logdet)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7)}

if __name__ == '__main__':
    sinesdata = AudibleSines()
    model = GoodSines()
    trainer = pl.Trainer()
    trainer.fit(model, sinesdata)