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

"""A copy and modified version of compressai/utils/bench/codecs.py"""

import abc
import io
import os
import platform
import subprocess
import sys
import time

from tempfile import mkstemp
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import PIL.Image as Image
import torch

from pytorch_msssim import ms_ssim

from compressai.transforms.functional import rgb2ycbcr, ycbcr2rgb

import lpips

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size


def read_image(filepath: str, mode: str = "RGB") -> np.array:
    """Return PIL image in the specified `mode` format."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return Image.open(filepath).convert(mode)


def _compute_psnr(a, b, max_val: float = 255.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def _compute_ms_ssim(a, b, max_val: float = 255.0) -> float:
    return ms_ssim(a, b, data_range=max_val).item()

def _compute_lpips(a, b, net='vgg'):
    loss_fn = lpips.LPIPS(net=net)
    return loss_fn(a, b).item()


_metric_functions = {
    "psnr-rgb": _compute_psnr,
    "ms-ssim-rgb": _compute_ms_ssim,
    "lpips-rgb": _compute_lpips,
}


def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    metrics: Optional[List[str]] = None,
    max_val: float = 255.0,
) -> Dict[str, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`."""

    if metrics is None:
        metrics = ["psnr-rgb"]

    def _convert(x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        x = torch.from_numpy(x.copy()).float().unsqueeze(0)
        if x.size(3) == 3:
            # (1, H, W, 3) -> (1, 3, H, W)
            x = x.permute(0, 3, 1, 2)
        return x

    a = _convert(a)
    b = _convert(b)

    out = {}
    for metric_name in metrics:
        if metric_name == "lpips-rgb":
            # LPIPS expects input in range [-1, 1]
            a_lpips = a / 127.5 - 1
            b_lpips = b / 127.5 - 1
            out[metric_name] = _metric_functions[metric_name](a_lpips, b_lpips)
        else:
            out[metric_name] = _metric_functions[metric_name](a, b, max_val)
    return out

def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)

class Codec(abc.ABC):
    """Abstract base class"""

    _description = None

    def __init__(self, args):
        self._set_args(args)

    def _set_args(self, args):
        return args

    @classmethod
    @abc.abstractmethod
    def setup_args(cls, _parser):
        pass

    @property
    def description(self):
        return self._description

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    def _load_img(self, img):
        return read_image(os.path.abspath(img))

    @abc.abstractmethod
    def _run_impl(self, img, quality, *args, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        in_filepath,
        quality: int,
        metrics: Optional[List[str]] = None,
        return_rec: bool = False,
    ):
        if metrics is None:
            metrics = ["psnr-rgb", "ms-ssim-rgb", "lpips-rgb"]
        info, rec = self._run_impl(in_filepath, quality)
        info.update(compute_metrics(rec, self._load_img(in_filepath), metrics))
        if return_rec:
            return info, rec
        return info

def get_vtm_encoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "EncoderApp", "Linux": "EncoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err


def get_vtm_decoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "DecoderApp", "Linux": "DecoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err


class VTM(Codec):
    """VTM: VVC reference software"""

    fmt = ".bin"

    @property
    def description(self):
        return "VTM"

    @property
    def name(self):
        return "VTM"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-b",
            "--build-dir",
            type=str,
            required=True,
            help="VTM build dir",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            required=True,
            help="VTM config file",
        )
        parser.add_argument(
            "--rgb", action="store_true", help="Use RGB color space (over YCbCr)"
        )

    def _set_args(self, args):
        args = super()._set_args(args)
        self.encoder_path = get_vtm_encoder_path(args.build_dir)
        self.decoder_path = get_vtm_decoder_path(args.build_dir)
        self.config_path = args.config
        self.rgb = args.rgb
        return args

    def _run_impl(self, in_filepath, quality):
        if not 0 <= quality <= 63:
            raise ValueError(f"Invalid quality value: {quality} (0,63)")

        # Taking 8bit input for now
        bitdepth = 8

        # Convert input image to yuv 444 file
        arr = np.asarray(self._load_img(in_filepath))
        fd, yuv_path = mkstemp(suffix=".yuv")
        out_filepath = os.path.splitext(yuv_path)[0] + ".bin"

        arr = arr.transpose((2, 0, 1))  # color channel first

        if not self.rgb:
            # convert rgb content to YCbCr
            rgb = torch.from_numpy(arr.copy()).float() / (2**bitdepth - 1)
            arr = np.clip(rgb2ycbcr(rgb).numpy(), 0, 1)
            arr = (arr * (2**bitdepth - 1)).astype(np.uint8)

        with open(yuv_path, "wb") as f:
            f.write(arr.tobytes())

        # Encode
        height, width = arr.shape[1:]
        cmd = [
            self.encoder_path,
            "-i",
            yuv_path,
            "-c",
            self.config_path,
            "-q",
            quality,
            "-o",
            "/dev/null",
            "-b",
            out_filepath,
            "-wdt",
            width,
            "-hgt",
            height,
            "-fr",
            "1",
            "-f",
            "1",
            "--InputChromaFormat=444",
            "--InputBitDepth=8",
            "--ConformanceWindowMode=1",
        ]

        if self.rgb:
            cmd += [
                "--InputColourSpaceConvert=RGBtoGBR",
                "--SNRInternalColourSpace=1",
                "--OutputInternalColourSpace=0",
            ]
        start = time.time()
        run_command(cmd)
        enc_time = time.time() - start

        # cleanup encoder input
        os.close(fd)
        os.unlink(yuv_path)

        # Decode
        cmd = [self.decoder_path, "-b", out_filepath, "-o", yuv_path, "-d", 8]
        if self.rgb:
            cmd.append("--OutputInternalColourSpace=GBRtoRGB")

        start = time.time()
        run_command(cmd)
        dec_time = time.time() - start

        # Compute PSNR
        rec_arr = np.fromfile(yuv_path, dtype=np.uint8)
        rec_arr = rec_arr.reshape(arr.shape)

        arr = arr.astype(np.float32) / (2**bitdepth - 1)
        rec_arr = rec_arr.astype(np.float32) / (2**bitdepth - 1)
        if not self.rgb:
            arr = ycbcr2rgb(torch.from_numpy(arr.copy())).numpy()
            rec_arr = ycbcr2rgb(torch.from_numpy(rec_arr.copy())).numpy()

        bpp = filesize(out_filepath) * 8.0 / (height * width)

        # Cleanup
        os.unlink(yuv_path)
        os.unlink(out_filepath)

        rec = Image.fromarray(
            (rec_arr.clip(0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        )

        # Save the reconstructed image
        rec_filepath = os.path.splitext(in_filepath)[0] + "_rec.png"
        rec.save(rec_filepath)    
        print(f'Saved reconstructed image to {rec_filepath}')

        out = {
            "bpp": bpp,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }

        return out, rec