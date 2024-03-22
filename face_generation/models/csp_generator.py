import torch
import torch.nn as nn


class CSPGenerator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.first_deconv = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.cspup_block = nn.Sequential(
            CSPUPBlock(ngf * 4),
            CSPUPBlock(ngf * 2),
            CSPUPBlock(ngf),
            CSPUPBlock(int(ngf / 2)),
        )
        self.outer_layer = nn.Sequential(
            nn.ConvTranspose2d(int(ngf / 2), nc, 4, 2, 1, bias=False), nn.ReLU(True)
        )

    def forward(self, x):
        x = self.first_deconv(x)
        x = self.cspup_block(x)
        x = self.outer_layer(x)
        return x


class CSPUPBlock(nn.Module):
    def __init__(self, nfg1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nfg1, nfg1, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                nfg1, nfg1, kernel_size=4, stride=2, padding=1, bias=False
            ),
            # # state ``nfg1, 8,8,
            #
            nn.Conv2d(nfg1, int(nfg1 / 2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(int(nfg1 / 2), nfg1, kernel_size=3, padding=1, bias=False),
        )
        self.deconv = nn.ConvTranspose2d(
            nfg1, nfg1, kernel_size=4, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        split_feature_maps = torch.split(x, int(x.size(1) / 2), dim=1)
        feature_map_1 = split_feature_maps[1]
        feature_map_0 = split_feature_maps[0]
        feature_map_1_after = self.main(feature_map_1)
        feature_map_0_after = self.deconv(feature_map_0)
        result = feature_map_0_after + feature_map_1_after
        return result


if __name__ == "__main__":
    fixed_noise = torch.randn(
        64,
        100,
        1,
        1,
    )
    t = CSPGenerator(100, 128, 3)(fixed_noise)
    print("___")
    print(t.shape)
