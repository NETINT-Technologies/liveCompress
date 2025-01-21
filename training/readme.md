# Training LIVE compression model

First, please ensure general setup steps in the root directory `README.md` file are followed, to ensure all dependencies are installed.

Then, cd into `train` and follow the instructions there to download the vimeo-90k-triplets dataset.

Once the dataset is downloaded, you can run the training script with the following command (please replace placeholder values for arguments with your desired values):

```bash
python main.py --d train --lambda {lambda_val} --batch_size {batch_size} --epochs {epochs} --lr {lr}
```

Please see the `config.py` file for more details on the arguments, additional arguments, and default values.