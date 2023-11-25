from typing import Any

import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import PeakSignalNoiseRatio
from src.utils.rgb_utils import create_montage, mask_image_torch
import torch.nn.functional as F
from torchmetrics.functional.image import total_variation

import numpy as np
import cv2
import os


class Noise2NoiseModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_type: str = "l1",
        compile=False,
        recon=False,
        tv_weight = 0,
        save_mask = True,
        time_test = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.recon = recon
        self.tv_weight = tv_weight
        self.save_mask = save_mask
        self.time_test = time_test

        # self.net = torch.compile(net) if compile else net
        self.net = torch.compile(net) if compile else net

        # loss function
        if self.hparams.loss_type.lower() == 'l2':
            self.loss = nn.MSELoss()
        elif self.hparams.loss_type.lower() == "l1":
            self.loss = nn.L1Loss()
        elif self.hparams.loss_type.lower() == "smoothl1":
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError("Loss not implemented.")

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_psnr = PeakSignalNoiseRatio()
        self.test_psnr = PeakSignalNoiseRatio()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_psnr_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_psnr.reset()
        self.val_psnr_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        # x = self.padtesttensor(x)
        # y = self.padtesttensor(y)
        restored_x = self.forward(x)
        loss = self.loss(restored_x, y)
        return loss, restored_x, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        if self.recon and self.current_epoch > 10:
            targets = self.forward(targets)
            loss += self.loss(targets, preds)
        if self.tv_weight > 0:
            loss += self.tv_weight * total_variation(preds, reduction='mean')
        
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # print(self.logger.log_dir)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_psnr(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        psnr = self.val_psnr.compute()  # get current val acc
        self.val_psnr_best(psnr)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/psnr_best", self.val_psnr_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_start(self):
        if self.time_test:
            import time


            # 生成随机数据
            data = torch.randn(128, 3, 640, 480)
            data = data.to(self.device)

            # 测量推理时间
            start_time = time.time()

            with torch.no_grad():
                for i in range(data.size(0)):
                    input = data[i].unsqueeze(0)  # 添加批次维度
                    predictions = self.forward(input)

            end_time = time.time()
            total_time = end_time - start_time
            print(f"处理20个数据点的推理总耗时: {total_time} 秒")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(os.path.join((self.logger.log_dir + "/"), f"denoised.mp4"), fourcc, 10.0, (640, 480))


    def on_test_end(self):
        self.out.release()

    def test_step(self, batch: Any, batch_idx: int):
        # force batch_size = 1 and no crop no redux
        x,_ = batch
        loss, preds, targets = self.model_step((x,_))
        # preds = self.forward(preds)
        x = x.squeeze(0)
        original_output = preds.clone().squeeze(0)
        preds = preds.squeeze(0)
        targets = targets.squeeze(0)
        mask = mask_image_torch(preds, threshold=20)
        if self.save_mask:
            import matplotlib.pyplot as plt

            # 假设你的 mask 是一个形状为 (3, 480, 640) 的 torch.Tensor
            # 取第一个通道
            mask_channel1 = mask[0, :, :]

            # 将 torch.Tensor 转换为 numpy 数组
            mask_np = mask_channel1.cpu().numpy()

            # 可视化
            plt.imshow(mask_np, cmap='gray')
            plt.axis('off')  # 关闭坐标轴
            plt.savefig(self.logger.log_dir + "/" + str(batch_idx) + "mask.png")  # 保存可视化的图像到本地
            plt.close()

        preds = x.clone()
        preds[mask] = 0

        denoised_image = create_montage(img_name=(str(batch_idx) + ".png"), noise_type="gaussian", save_path=self.logger.log_dir + "/", source_t=x, denoised_t=preds, clean_t=original_output, show=0)

        self.out.write(np.array(denoised_image))

        # update and log metrics
        self.test_loss(loss)
        self.test_psnr(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/psnr", self.test_psnr, on_step=False, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self):
        pass

    def padtesttensor(self, input):
        original_height = input.size(2)
        original_width = input.size(3)

        # 计算长边尺寸
        longer_side = max(original_height, original_width)

        # 计算目标尺寸，使其与长边尺寸相同
        target_height = target_width = longer_side

        # 计算需要进行填充的数量
        pad_height = max(0, target_height - original_height)
        pad_width = max(0, target_width - original_width)

        # 计算填充量
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad

        # 使用零填充调整尺寸
        padded_input = F.pad(input, (left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=0)
        return padded_input

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Noise2NoiseModule(None, None, None)
