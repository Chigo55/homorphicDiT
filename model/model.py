import torch
import torch.nn as nn
import lightning as L

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from model.losses import *
from model.block import *
from utils.metrics import *
from utils.hook import add_nan_hooks


class HomomorphicUnet(nn.Module):
    def __init__(self, image_size, offset, cutoff):
        super().__init__()

        self.rgb2ycrcb = RGB2YCrCb(
            offset=offset
        )
        self.homo_separate = HomomorphicSeparation(
            size=image_size,
            cutoff=cutoff
        )
        self.unet = UNet(
        )
        self.refine = IterableRefine(
        )
        self.ycrcb2rgb = YCrCb2RGB(
            offset=offset
        )

    def forward(self, x):
        Y, Cr, Cb = self.rgb2ycrcb(x)
        x_i, x_d = self.homo_separate(Y)
        o_i = self.unet(x_i)           # 보정 계수
        n_i = self.refine(x_i, o_i)
        n_Y = torch.clamp(input=n_i * x_d, min=0.0, max=1.0)
        enh_img = self.ycrcb2rgb(n_Y, Cr, Cb)
        return Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img


class HomomorphicUnetLightning(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = HomomorphicUnet(
            image_size=hparams['image_size'],
            offset=hparams['offset'],
            cutoff=hparams["cutoff"],
        )
        add_nan_hooks(module=self)

        self.spa_loss = L_spa().eval()
        self.col_loss = L_col().eval()
        self.exp_loss = L_exp().eval()
        self.tva_loss = L_tva().eval()

        self.lambda_spa = hparams["lambda_spa"]
        self.lambda_col = hparams["lambda_col"]
        self.lambda_exp = hparams["lambda_exp"]
        self.lambda_tva = hparams["lambda_tva"]

        self.metric = ImageQualityMetrics(device="cuda")
        self.metric.eval()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)

        loss_spa = self.lambda_spa * torch.mean(
            input=self.spa_loss(enh_img, x)
        )
        loss_col = self.lambda_col * torch.mean(
            input=self.col_loss(enh_img)
        )
        loss_exp = self.lambda_exp * torch.mean(
            input=self.exp_loss(enh_img)
        )
        loss_tva = self.lambda_tva * torch.mean(
            input=self.tva_loss(o_i)
        )

        loss_tot = (
            loss_spa +
            loss_col +
            loss_exp +
            loss_tva
        )

        self.log_dict(dictionary={
            "train/1_spa": loss_spa,
            "train/2_col": loss_col,
            "train/3_exp": loss_exp,
            "train/4_tva": loss_tva,
            "train/5_tot": loss_tot,
        }, prog_bar=True)

        if torch.isnan(input=loss_spa):
            print("LOSS SPA IS NAN!")
        if torch.isnan(input=loss_col):
            print("LOSS COL IS NAN!")
        if torch.isnan(input=loss_exp):
            print("LOSS EXP IS NAN!")
        if torch.isnan(input=loss_tva):
            print("LOSS TVA IS NAN!")
        if torch.isnan(input=loss_tot):
            print("LOSS TOT IS NAN!")

        if batch_idx % 50 == 0:
            self.logger.experiment.add_images(
                "train/1_input",
                x,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/2_Y",
                Y,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/3_Cr",
                Cr,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/4_Cb",
                Cb,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/5_x_i",
                x_i,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/6_x_d",
                x_d,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/7_o_i",
                o_i,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/8_n_i",
                n_i,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/9_n_Y",
                n_Y,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/0_enh_img",
                enh_img,
                self.global_step
            )
        return loss_tot

    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)

        loss_spa = self.lambda_spa * torch.mean(
            input=self.spa_loss(enh_img, x)
        )
        loss_col = self.lambda_col * torch.mean(
            input=self.col_loss(enh_img)
        )
        loss_exp = self.lambda_exp * torch.mean(
            input=self.exp_loss(enh_img)
        )
        loss_tva = self.lambda_tva * torch.mean(
            input=self.tva_loss(o_i)
        )

        loss_tot = (
            loss_spa +
            loss_col +
            loss_exp +
            loss_tva
        )

        self.log_dict(dictionary={
            "valid/1_spa": loss_spa,
            "valid/2_col": loss_col,
            "valid/3_exp": loss_exp,
            "valid/4_tva": loss_tva,
            "valid/5_tot": loss_tot,
        }, prog_bar=True)
        return loss_tot

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)

        metrics = self.metric.full(preds=enh_img, targets=x)

        self.log_dict(dictionary={
            "bench/1_PSNR": metrics["PSNR"],
            "bench/2_SSIM": metrics["SSIM"],
            "bench/3_LPIPS": metrics["LPIPS"],
            "bench/4_BRISQUE": metrics["BRISQUE"],
        }, prog_bar=True)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)
        return enh_img

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['decay'],
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=10,           # 첫 번째 주기의 epoch 수
            T_mult=2,         # 이후 주기의 길이 배수
            eta_min=1e-8      # 최소 learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   # 매 epoch마다 업데이트
                "frequency": 1,
            }
        }
