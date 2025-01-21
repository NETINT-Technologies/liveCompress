from dataclasses import dataclass
from typing import Tuple, Optional
import argparse

from compressai.zoo import image_models  # This will import all models directly

@dataclass
class TrainingConfig:
    # All fields required for model training along with their default values
    model: str = "liveModel"
    dataset: str = ""  # Required, so no default
    epochs: int = 100
    learning_rate: float = 1e-4
    num_gpus: int = 1
    test_size: int = 500
    train_size: int = 10000
    num_workers: int = 4
    lmbda: float = 1e-2
    beta: float = 5e-3
    filename: str = "checkpoint"
    batch_size: int = 16
    test_batch_size: int = 64
    aux_learning_rate: float = 1e-3
    patch_size: Tuple[int, int] = (256, 256)
    cuda: bool = False
    seed: Optional[int] = None
    clip_max_norm: float = 1.0
    checkpoint: Optional[str] = None
    ae_dataset: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'TrainingConfig':
        # Only use non-None values from args to override default values
        args_dict = {k: v for k, v in vars(args).items() if v is not None}
        return cls(**args_dict)

    @classmethod
    def get_default(cls) -> 'TrainingConfig':
        """Create a config with default values"""
        return cls()

"""
Parses the arguments from the command line and returns a TrainingConfig object.
If an argument is not provided, the default values defined above are used.
"""
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="liveModel",
        choices=image_models.keys(),
        help="Model architecture",
    )
    parser.add_argument(
        "-d", 
        "--dataset", 
        type=str, 
        required=True, 
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "-g",
        "--num_gpus",
        type=int,
        help="Number of GPUs to use for training",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="Number of folders to use for testing",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        help="Number of folders to use for training",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        help="Dataloaders threads",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        help="Bit-rate distortion parameter",
    )
    parser.add_argument(
        "--beta",
        dest="beta",
        type=float,
        help="Latent weight parameter",
    )
    parser.add_argument(
        "--filename",
        dest="filename",
        type=str,
        default="checkpoint",
        help="Checkpoint prefix filename (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        help="Test batch size",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        help="Auxiliary loss learning rate",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        help="Size of the patches to be cropped",
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Use cuda"
    )
    parser.add_argument(
        "--seed", type=int, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        type=float,
        help="gradient clipping max norm",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to a checkpoint"
    )
    parser.add_argument(
        "-ae", "--ae_dataset", action="store_true", help="Whether to use an autoencoder dataset"
    )
    
    args = parser.parse_args(argv)
    return TrainingConfig.from_args(args)