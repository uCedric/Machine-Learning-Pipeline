import matplotlib.pyplot as plt
from .logger import setup_logger

logger = setup_logger("Plotter")

def plot_history(history, output_path="loss_chart.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    logger.info(f"Loss chart saved to {output_path}")
