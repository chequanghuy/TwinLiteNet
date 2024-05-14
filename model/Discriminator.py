
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim=8192, hidden_dim=256, num_domains=1):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Conv2d(2, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Flatten(),


            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        pass
        return self.layers(x)

if __name__ == "__main__" :
    x = torch.randn([16,2,64,64])
    # x = x.view(x.size(0), -1)
    net = Discriminator(input_dim=x[0].size(0))




    p = net(x)
    print(p.shape)