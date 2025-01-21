# Testing scripts

This folder contains various scripts that can be used for evaluating the performance of trained models on the kodak dataset. There are four main scripts here:

### `eval_single_image.py`
`eval_single_image.py` takes one image from the `../kodak` folder whose path is specified in a command line argument, and one trained checkpoint from the `../checkpoints` folder whose path is also specified through the command line. It passes the image through the compression model loaded from the checkpoint, saving it at various stages, namely the latent, precomp (decoded directly from latent), and compressed (result after running through model). It also runs some traditional compression algorithms (jpeg and webp) on the decoded latent and stores the resulting images to compare to our LIVE model. Then, bpp is calculated and displayed for all approaches (LIVE and the traditional compression algorithms).

Example usage: ```python eval_single_image.py --checkpoint checkpoint_1e-3_best_loss.pth.tar --image kodim09.png```

### `eval_single_checkpoint.py`
`eval_single_checkpoint.py` takes one checkpoint from the `../checkpoints` folder whose path is specified through the command line. It runs all 24 images in the kodak dataset through the compression model loaded from the checkpoint, saving the image at various stages (latent, precomp, and live-compressed). On the decoded latent, further traditional compression algorithms are run and the resulting images are also stored. For the traditional algorithms and LIVE, bpp, lpips, and psnr metrics are calculated for each individual image and then the average is taken and displayed. Finally, the metric result for each image and method is plotted on a scatterplot of BPP vs LPIPS and BPP vs PSNR.

Example usage: ```python eval_single_checkpoint.py --checkpoint checkpoint_1e-3_best_loss.pth.tar```

### `get_csv_of_test_metrics.py`
`get_csv_of_test_metrics.py` runs a script that sweeps through all checkpoints in the `../checkpoints` folder, loading the corresponding model for each checkpoint. It then takes all 24 images in the kodak dataset and passes each of them through the model, saving them at various stages (latent, precomp, live-compressed), and calculating all relevant metrics (bpp, psnr, lpips). On the decoded latent, jpeg, heif, and webp algorithms are run and the resulting metrics are calculated for comparison with the LIVE results. After going through all checkpoints, an `.xlsx` file with three sheets, one for each of bpp, psnr, and lpips is generated, containing a table of the metric result for each image trained using each checkpoint, along with the results for each image's jpeg, heif, and webp compressions. Individual `.csv` files for each metric are also generated for backup purposes.

Example usage: ```python get_csv_of_test_metrics.py```

### `sweep_all.ipynb`
`sweep_all.ipynb` offers very similar functionality to `get_csv_of_test_metrics.py` except results at intermediate phases can be seen and visualized. Furthermore, a number of plots are also generated, namely a bpp vs psnr and bpp vs lpips curve for LIVE and each of the traditional compression algorithms. Different from `get_csv_of_test_metrics.py`, the performance of the encoding and decoding for LIVE is also calculated and compared with traditional algorithms, and visualized in another bar chart. More specifically, comparison between the VAE autoencoder encode and decode times and that of LIVE can also be visualized.

To run this, open the file up in jupyter notebook and click on `run all cells`, or run each individual cell. Ensure you have selected the correct kernel with all required dependencies installed.

### Additional instructions

- Please ensure you have the kodak dataset downloaded. Refer to instructions found in `../kodak` if needed.
- Please ensure all required dependencies from `../requirements.txt` are installed.
- For first time setup, please run the training script first and copy the generated checkpoints into the `../checkpoints` folder prior to running any of these scripts. See instructions in the training folder for more details.

