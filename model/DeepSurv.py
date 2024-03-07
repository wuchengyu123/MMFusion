import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class DeepSurv(nn.Module):
    def __init__(self,in_dim):
        super(DeepSurv, self).__init__()
        self.deepsurv = nn.Sequential(
            nn.Linear(in_dim,64),
            nn.BatchNorm1d(64),
            nn.ELU(alpha=0.5),
            nn.Dropout(p=0.22),
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(p=0.22),
            nn.Linear(128,768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),
            nn.Dropout(p=0.22)
        )

        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,clinical_features):
        out3 = self.deepsurv(clinical_features)
        return out3


if __name__ == '__main__':
    x1 = torch.rand((256,54))
    model = DeepSurv()
    output = model(x1)
    print(output.shape)