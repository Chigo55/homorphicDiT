import torch
import torch.nn as nn
import torch.nn.functional as F


class RGB2YCrCb(nn.Module):
    def __init__(self, offset=0.5,):
        super().__init__()
        self.offset = float(offset)

    def forward(self, x):
        R = x[:, 0:1, :, :].clone()
        G = x[:, 1:2, :, :].clone()
        B = x[:, 2:3, :, :].clone()

        offset = x.new_tensor(self.offset)

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + offset
        Cb = (B - Y) * 0.5256 + offset

        Y = torch.clamp(input=Y, min=0.0, max=1.0)
        Cr = torch.clamp(input=Cr, min=0.0, max=1.0)
        Cb = torch.clamp(input=Cb, min=0.0, max=1.0)
        return Y, Cr, Cb


class YCrCb2RGB(nn.Module):
    def __init__(self, offset=0.5,):
        super().__init__()
        self.offset = float(offset)

    def forward(self, Y, Cr, Cb):
        offset = Y.new_tensor(self.offset)

        Cr = Cr - offset
        Cb = Cb - offset

        R = Y + 1.403 * Cr
        G = Y - 0.344 * Cb - 0.714 * Cr
        B = Y + 1.773 * Cb

        RGB = torch.cat(tensors=[R, G, B], dim=1)
        RGB = torch.clamp(input=RGB, min=0, max=1)
        return RGB


class HomomorphicSeparation(nn.Module):
    def __init__(self, size=256, cutoff=0.1, trainable=False, eps=1e-6):
        super().__init__()
        self.size = size
        self.eps = float(eps)

        # cutoff → logit 로 저장해 0~1 범위 보장
        p = torch.tensor(data=cutoff, dtype=torch.float64)
        logit = torch.log(input=p / (1.0 - p))
        self.raw_cutoff = nn.Parameter(logit, requires_grad=trainable)

        # 고정 좌표계
        coord = torch.linspace(start=-1.0, end=1.0, steps=size)
        y, x = torch.meshgrid(coord, coord, indexing="ij")      # (H, W)
        # (H, W) on CPU float64
        self.radius = torch.sqrt(input=x**2 + y**2)
    # ------------------------------------------------------------

    def _gaussian_lpf(self, dtype, device) -> torch.Tensor:
        cutoff = torch.sigmoid(input=self.raw_cutoff).to(
            dtype=dtype, device=device)  # scalar
        radius = self.radius.to(
            dtype=dtype, device=device)                     # (H, W)
        h = torch.exp(input=-(radius**2) / (2.0 * cutoff**2))        # (H, W)
        return h                                          # (H,W)

    # ------------------------------------------------------------
    def forward(self, x):
        dtype, device = x.dtype, x.device
        b = x.shape[0]

        # 1. 로그 공간
        x_log = torch.log(input=x + self.eps)

        # 2. FFT → 주파수 영역
        x_fft = torch.fft.fft2(x_log, norm='ortho')
        x_fft = torch.fft.fftshift(x_fft)

        # 3. 필터링
        H = self._gaussian_lpf(dtype=dtype, device=device)  # (H,W)
        H = H.unsqueeze(dim=0).unsqueeze(
            dim=0)                           # (1,1,H,W)

        low_fft = x_fft * H               # 저주파
        high_fft = x_fft * (1 - H)         # 고주파

        # 4. 각각 iFFT
        low_ifft = torch.fft.ifft2(
            torch.fft.ifftshift(low_fft),
            norm='ortho'
        ).real
        high_ifft = torch.fft.ifft2(
            torch.fft.ifftshift(high_fft),
            norm='ortho'
        ).real

        # 5. 지수 복원
        low = torch.exp(input=low_ifft) - self.eps
        high = torch.exp(input=high_ifft) - self.eps

        # 6. 정규화 및 출력
        low = torch.clamp(input=low,  min=0.0, max=1.0)
        high = torch.clamp(input=high, min=0.0, max=1.0)
        return low, high


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = (in_channels + out_channels)/2
        hidden_channels = int(hidden_channels)

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode='replicate'
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.SiLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                padding_mode='replicate'
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            DoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
            )
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        x = torch.clamp(input=x, min=0.0, max=1.0)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]

        x1_pad = F.pad(
            input=x1_up,
            pad=[
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2
            ]
        )
        x = torch.cat(tensors=[x2, x1_pad], dim=1)
        x = self.conv(x)
        x = torch.clamp(input=x, min=0.0, max=1.0)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.inc = DoubleConv(in_channels=1, out_channels=8)
        self.down1 = Down(in_channels=8, out_channels=16)
        self.down2 = Down(in_channels=16, out_channels=32)
        self.down3 = Down(in_channels=32, out_channels=64)
        self.down4 = Down(in_channels=64, out_channels=128)
        self.up1 = Up(in_channels=128, out_channels=64)
        self.up2 = Up(in_channels=64, out_channels=32)
        self.up3 = Up(in_channels=32, out_channels=16)
        self.up4 = Up(in_channels=16, out_channels=8)
        self.outc = self.outc = DoubleConv(in_channels=8, out_channels=1)

    def forward(self, x):
        x_i = self.inc(x)           # 8
        d_1 = self.down1(x_i)       # 16
        d_2 = self.down2(d_1)       # 32
        d_3 = self.down3(d_2)       # 64
        d_4 = self.down4(d_3)       # 1024
        u_4 = self.up1(d_4, d_3)    # 64
        u_3 = self.up2(u_4, d_2)    # 32
        u_2 = self.up3(u_3, d_1)    # 16
        u_1 = self.up4(u_2, x_i)    # 8
        x_o = self.outc(u_1)
        x_o = torch.clamp(input=x_o, min=0.0, max=1.0)
        return x_o


class IterableRefine(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            DoubleConv(in_channels=1, out_channels=8),
            nn.Sigmoid(),
        )

    def forward(self, x, r):
        r = torch.clamp(input=self.block(r), min=0.0, max=1.0)
        # delta = x.pow(2) + x    # (B,1,H,W)
        delta = x * (x + 1e-4)    # (B,1,H,W)
        refined = x + (r * delta).sum(dim=1, keepdim=True)
        refined = torch.clamp(input=refined, min=0.0, max=1.0)
        return refined
