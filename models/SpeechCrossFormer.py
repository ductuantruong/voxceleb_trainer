from vit_pytorch.crossformer import CrossFormer

import torch
import torch.nn as nn

import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch.nn.functional as F
from utils import PatchEmbed

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(
            inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        return F.conv1d(inputs, self.flipped_filter).squeeze(1)


class Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop=160, n_mels=80, coef=0.97, requires_grad=False):
        super(Mel_Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop = hop

        self.pre_emphasis = PreEmphasis(coef)
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.mel_basis = nn.Parameter(
            torch.FloatTensor(mel_basis), requires_grad=requires_grad)
        self.instance_norm = nn.InstanceNorm1d(num_features=n_mels)
        window = torch.hamming_window(self.win_length)
        self.window = nn.Parameter(
            torch.FloatTensor(window), requires_grad=False)

    def forward(self, x, x_len=None):
        x = self.pre_emphasis(x)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                       window=self.window, win_length=self.win_length, return_complex=True)
        x = torch.abs(x)
        x += 1e-9
        x = torch.log(x)
        x = torch.matmul(self.mel_basis, x)
        x = self.instance_norm(x)
        x = x.unsqueeze(1)
        return x

class FBanks(nn.Module):
    def __init__(self, n_fft=512, window_size=400, hop_size=160, mel_bins=128, 
                 window='hann', center=True, pad_mode='reflect', sr=16000, fmin=50, fmax=14000):
        super(FBanks, self).__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sr, n_fft=n_fft, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None, 
            freeze_parameters=True)
        self.bn = nn.BatchNorm2d(mel_bins)

    def forward(self, x):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x) # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn(x)
        x = x.transpose(1, 3)
        return x
    
class SpeechCrossFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_trans = FBanks()
        self.encoder = CrossFormer(
                            num_classes = 96,                # number of output classes
                            dim = (32, 64, 128, 256),         # dimension at each stage
                            depth = (2, 2, 8, 2),              # depth of transformer at each stage
                            global_window_size = (4, 2, 2, 1), # global window sizes at each stage
                            local_window_size = 4,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
                        )
        # self.encoder = dnn_34(num_classes = 1)
        self.patch_emb = PatchEmbed()
    
    def forward(self, x):
        feature = self.mel_trans(x) # to compute fbank feature
        feature = self.patch_emb(feature)
        embedding = self.encoder(feature) # feature extractor to extract speaker embedding
        return embedding
    
def MainModel(**kwargs):
    model = SpeechCrossFormer()
    return model
