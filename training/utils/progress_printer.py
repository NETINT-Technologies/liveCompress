from multiprocessing import Manager

class ProgressPrinter:
    """Prints progress of training and testing accross all processes (GPUs)
    
    Args:
        total_size (int): Total number of samples in the dataset.
        batch_size (int): Number of samples per batch.
    """
    
    def __init__(self, batch_size: int, manager: Manager):
        self.batch_size = batch_size
        self.num_complete_batch = manager.Value('i', 0)  # Shared integer value
        self.total_size = None
        self.lock = manager.Lock()  # Shared lock
        self.max_train_time = manager.Value('d', 0.0)
    
    def set_total_size(self, total_size: int):
        """Set the total number of samples in the dataset."""
        self.total_size = total_size
        
    def reset(self):
        """Reset the number of completed batches."""
        with self.lock:
            self.num_complete_batch.value = 0
    
    def print_progress(self, epoch, loss, latent_kl_loss, mse_loss, bpp_loss, aux_loss, logger):
        """For every 10 batches that completes training, print the progress including all losses."""
        with self.lock:
            self.num_complete_batch.value += 1
            current_batch = self.num_complete_batch.value

        if current_batch % 10 == 0:
            print(f"Total size: {self.total_size}")
            logger.info(
                f"Train epoch {epoch}: ["
                f"{current_batch*self.batch_size}/{self.total_size}"
                f" ({100. * current_batch * self.batch_size / self.total_size:.0f}%)]"
                f'\tLoss: {loss:.4f} |'
                f'\tLatent Loss: {latent_kl_loss:.5f} |'
                f'\tMSE loss: {mse_loss:.5f} |'
                f'\tBpp loss: {bpp_loss:.3f} |'
                f"\tAux loss: {aux_loss:.2f}"
            )
            
    def update_max_train_time(self, train_time: float):
        """Update the maximum training time across all processes."""
        with self.lock:
            if train_time > self.max_train_time.value:
                self.max_train_time.value = train_time

    def get_max_train_time(self):
        return self.max_train_time.value