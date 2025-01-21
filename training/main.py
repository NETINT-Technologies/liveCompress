import os
import sys
import time
import random
import warnings
from multiprocessing import Manager

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from diffusers.models import AutoencoderKL

from config import TrainingConfig, parse_args
from train import Train
from data_loader import DataModule
from utils import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import liveModel 
from model.rate_distortion_custom import CustomRateDistortionLoss

#setup logging
logger = setup_logging(__name__, level="INFO")

"""
Process spawned by main to go through the entire training pipeline, including loading the images, defining the model and autoencoder, training, and testing.
Args:
    rank: 0-indexed process id.
    shared_dict: A shared dictionary to store the ProgressPrinter.
    config: The training configuration loaded from either command line arguments or default values.
"""
def main_worker(rank, shared_dict, config: TrainingConfig):
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
    dist.init_process_group("nccl", rank=rank, world_size=config.num_gpus)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        random.seed(config.seed)

    # Get the shared progress_printer instance
    progress_printer = shared_dict['progress_printer']
    
    if rank == 0:  # Only set total size once from the main process
        progress_printer.set_total_size(config.train_size)

    data_module = DataModule(config)
    data_module.setup()
    
    train_dataloader = data_module.get_train_dataloader(rank)
    test_dataloader = data_module.get_test_dataloader(rank)
    
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    lpips_metric.to(rank)
    
    writer = SummaryWriter()

    net = liveModel(N=24, M=36).to(rank)
    net = DDP(net, device_ids=[rank])

    ae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    ae = ae.to(rank)
    ae.eval()

    criterion = CustomRateDistortionLoss(lmbda=config.lmbda, beta=config.beta, metric="latent-mix")
    
    trainer = Train(
        rank=rank,
        model=net,
        ae=ae,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizers=None,
        clip_max_norm=config.clip_max_norm,
        lpips=lpips_metric,
        config=config,
        writer=writer,
        logger=logger,
        progress_printer=progress_printer
    )

    # Use trainer's configure_optimizers method
    optimizer, aux_optimizer = trainer.configure_optimizers(net, config)
    optimizers = {
        "net": optimizer,
        "aux": aux_optimizer
    }
    # Update trainer with configured optimizers
    trainer.optimizers = optimizers
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    last_epoch = 0
    if config.checkpoint:
        logger.info(f"Loading {config.checkpoint}")
        checkpoint = torch.load(config.checkpoint, map_location=f'cuda:{rank}')
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
    best_loss = float("inf")
    for epoch in range(last_epoch, config.epochs):
        # Set epoch for the train dataloader's sampler to ensure proper shuffling
        train_dataloader.sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.debug(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            logger.debug(f"Lambda: {config.lmbda}")
            
        trainer.train_one_epoch(epoch=epoch)
        
        progress_printer.reset()
        start_time = time.time()
        
        loss = trainer.test_epoch(epoch=epoch)
        
        end_time = time.time()
        if rank == 0:
            logger.info(f"Epoch {epoch} testing time: {end_time - start_time:0.2f}")
        
        lr_scheduler.step(loss)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best,
            config.lmbda,
            config.filename
        )

def main(argv):
    config = parse_args(argv)
    
    # Create a manager to share objects between processes
    manager = Manager()
    # Create a shared dictionary to store the ProgressPrinter
    shared_dict = manager.dict()
    shared_dict['progress_printer'] = ProgressPrinter(config.batch_size, manager)
    
    mp.spawn(main_worker,
        args=(shared_dict, config),
        nprocs=config.num_gpus,
        join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main(sys.argv[1:])