from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

class SandData(Dataset):
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data.values).float()
        self.y = torch.tensor(y_data.values).float()
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        signal = self.X[index]
        signal = signal[~signal.isnan()]
        signal_len = len(signal)
        signal = signal.repeat(3000 // signal_len)
        
        num_missing_samples = 3000 - len(signal)
        last_dim_padding = (0, num_missing_samples)
        signal = F.pad(signal, last_dim_padding)
        
        return signal, self.y[index]


class SandDataTest(Dataset):
    def __init__(self, X_data):
        self.X = torch.tensor(X_data.values).float()

        # self.orig_sr = 117.2 * 1000
        # self.resamler = transforms.Resample(self.orig_sr, 16000)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        signal = self.X[index]
        signal = signal[~signal.isnan()]
        signal_len = len(signal)
        signal = signal.repeat(3000 // signal_len)

        num_missing_samples = 3000 - len(signal)
        last_dim_padding = (0, num_missing_samples)
        signal = F.pad(signal, last_dim_padding)

        return signal