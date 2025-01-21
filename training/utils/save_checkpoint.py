import torch
import shutil

"""
Saves a checkpoint of the model with a specified filename based on the lambda value.
"""
def save_checkpoint(state, is_best, lambda_val, filename):
    torch.save(state, filename + ".pth.tar")
    if is_best:
        try:
            # Convert to scientific notation with format like "1e-3"
            lambda_str = f"{lambda_val:.0e}".replace("0", "")
            shutil.copyfile(filename + ".pth.tar", filename + "_" + lambda_str + "_best_loss.pth.tar")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")