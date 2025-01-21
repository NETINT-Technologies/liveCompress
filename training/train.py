# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import math
import os

import time
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from compressai.optimizers import net_aux_optimizer

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import logging
from config import TrainingConfig
from utils import *

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

class Train:
    def __init__(
            self,
            rank: int,
            model: torch.nn.Module,
            ae: torch.nn.Module,
            criterion: torch.nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader,
            optimizers: Dict[str, torch.optim.Optimizer],
            clip_max_norm: float,
            lpips: LearnedPerceptualImagePatchSimilarity,
            config: TrainingConfig,
            writer: SummaryWriter, #should pass this in from main
            logger: logging.Logger, #should pass this in from main
            progress_printer: ProgressPrinter,
    ):
        self.rank = rank
        self.model = model
        self.ae = ae
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizers = optimizers
        self.clip_max_norm = clip_max_norm
        self.lpips = lpips 
        self.config = config
        self.writer = writer
        self.progress_printer = progress_printer
        self.logger = logger
        
    def configure_optimizers(self, net, args):
        """Separate parameters for the main optimizer and the auxiliary optimizer.
        Return two optimizers"""
        conf = {
            "net": {"type": "Adam", "lr": args.learning_rate},
            "aux": {"type": "Adam", "lr": args.aux_learning_rate},
        }
        optimizer = net_aux_optimizer(net, conf)
        return optimizer["net"], optimizer["aux"]


    def train_one_epoch(
        self,
        epoch: int
    ):
        epoch_size = len(self.train_dataloader.dataset)
        self.logger.info(f"Training epoch {epoch} with {epoch_size} samples")
        batch_size = self.train_dataloader.batch_size
        self.model.train()
        device = next(self.model.parameters()).device
        start_time = time.time()
        
        optimizer = self.optimizers["net"]
        aux_optimizer = self.optimizers["aux"]
        
        for i, d in enumerate(self.train_dataloader):
            d = d.to(device)

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            with torch.no_grad():
                ae_i = self.ae.encode(d).latent_dist 
            sample = ae_i.sample()
            out_net = self.model(sample)
            out_criterion = self.criterion(out_net, ae_i)
    
            if torch.isnan(out_criterion["loss"]):
                self.logger.warning(f"Warning: NaN loss encountered in epoch {epoch}, batch {i}, rank {self.rank}")
            
        
            out_criterion["loss"].backward()
            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
            optimizer.step()
            aux_loss = self.model.module.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            self.progress_printer.print_progress(epoch, out_criterion["loss"].item(), out_criterion["latent_kl_loss"].item(), out_criterion["mse_loss"].item(), out_criterion["bpp_loss"].item(), aux_loss.item(), self.logger)
        
            if i % 10 == 0:
                step = epoch * math.ceil(epoch_size / batch_size) / 10 + i / 10
                self.writer.add_scalar("Loss/train", out_criterion["loss"].item(), global_step=step)
                self.writer.add_scalar("MSE Loss/train", out_criterion["mse_loss"].item(), global_step=step)
                self.writer.add_scalar("KL Loss/train", out_criterion["latent_kl_loss"].item(), global_step=step)
                self.writer.add_scalar("Bpp Loss/train", out_criterion["bpp_loss"].item(), global_step=step)
                self.writer.add_scalar("Aux Loss/train", aux_loss.item(), global_step=step)
            
        with torch.no_grad():
            ae_o = self.ae.decode(out_net["x_hat"]).sample
            comp_o = self.ae.decode(ae_i.sample()).sample
            self.logger.debug(f"AE output shape: {ae_o.shape}, compressed output shape: {comp_o.shape}")
            lpips_loss = self.lpips(ae_o.clamp(0,1),comp_o.clamp(0,1))
        self.writer.add_scalar("LPIPS Loss/train",  lpips_loss.item(), global_step=epoch)
        self.writer.add_images("Original/train",  d.detach().cpu(), global_step=epoch)
        self.writer.add_images("Latent/train",  comp_o.detach().cpu(), global_step=epoch)
        self.writer.add_images("Reconstructed/train", ae_o.detach().cpu(), global_step=epoch)
        end_time = time.time()
        train_time = end_time - start_time
        self.progress_printer.update_max_train_time(train_time)

        # Synchronize all processes
        dist.barrier()

        # Only rank 0 prints the max training time
        if self.rank == 0:
            max_time = self.progress_printer.get_max_train_time()
            self.logger.info(f"Epoch {epoch} max training time: {max_time:.2f} seconds")

        # Another barrier to ensure print is complete before moving on
        dist.barrier()


    def test_epoch(
        self,
        epoch: int, 
    ):
        epoch_size = len(self.test_dataloader.dataset)
        batch_size = self.test_dataloader.batch_size
        self.model.eval()
        device = next(self.model.parameters()).device
        img_mse_loss_metric = nn.MSELoss()
        img_mse_loss_metric.to(device)

        loss = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        aux_loss = AverageMeter()
        lpips_loss = AverageMeter()
        img_mse_loss = AverageMeter()
        
        with torch.no_grad():
            for i, d in enumerate(self.test_dataloader):
                d = d.to(device)
                ae_i = self.ae.encode(d).latent_dist 
                out_net = self.model(ae_i.sample())
                out_criterion = self.criterion(out_net, ae_i)

                aux_loss.update(self.model.module.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                
            step = epoch * math.ceil(epoch_size / batch_size) + i

            with torch.no_grad():
                decoded_latent = self.ae.decode(out_net["x_hat"]).sample
                base_latent = self.ae.decode(ae_i.sample()).sample
                lpips_loss.update(self.lpips(decoded_latent.clamp(0,1), base_latent.clamp(0,1)))
            img_mse_loss.update(img_mse_loss_metric(decoded_latent, base_latent) )
            self.writer.add_scalar("Loss/test", loss.val, global_step=step)
            self.writer.add_scalar("MSE Loss/test", mse_loss.val, global_step=step)
            self.writer.add_scalar("Bpp Loss/test", bpp_loss.val, global_step=step)
            self.writer.add_scalar("Aux Loss/test", aux_loss.val, global_step=step)

            
        with torch.no_grad():
            decoded_latent = self.ae.decode(out_net["x_hat"]).sample
            base_latent = self.ae.decode(ae_i.sample()).sample
            lpips_loss.update(self.lpips(decoded_latent.clamp(0,1), base_latent.clamp(0,1)))
        img_mse_loss.update(img_mse_loss_metric(decoded_latent, base_latent) )
        self.writer.add_scalar("LPIPS Loss/test", lpips_loss.val, global_step=epoch)
        self.writer.add_scalar("Image Loss/test", img_mse_loss.val, global_step=epoch)
        self.writer.add_images("AE Decoded/test",  base_latent.detach().cpu(), global_step=epoch)
        self.writer.add_images("Reconstructed/test", decoded_latent.detach().cpu(), global_step=epoch)
    
        if self.rank == 0:
            self.logger.info(
                f"Test epoch {epoch}: Average losses:"
                f"\tLoss: {loss.avg:.4f} |"
                f"\tMSE loss: {mse_loss.avg:.4f} |"
                f"\tLPIPS Loss: {lpips_loss.avg.squeeze():.4f} |"
                f"\tImg MSE Loss: {img_mse_loss.avg:.4f} |"
            f"\tBpp loss: {bpp_loss.avg:.3f} |"
                f"\tAux loss: {aux_loss.avg:.2f}\n"
            )
        return loss.avg