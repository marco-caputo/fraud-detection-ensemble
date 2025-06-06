import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import *

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))


def save_training_error_plot(model_name: str, train_losses: list[float], val_losses: list[float]):
    """
    Saves training and validation loss plot using seaborn.
    This function creates a line plot to visualize the training and validation losses over epochs.
    :param model_name: the name of the model being trained, used for plot title
    :param train_losses: the list of training losses per epoch
    :param val_losses: the list of validation losses per epoch
    """
    # Prepare data
    loss_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses
    })
    loss_df = loss_df.melt(id_vars='Epoch', var_name='Type', value_name='Loss')

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=loss_df, x="Epoch", y="Loss", hue="Type", marker="o")
    plt.title(f"{model_name}: Training and Validation Loss over Epochs")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(script_dir, IMAGES_FOLDER_NAME, f"{model_name}_loss_plot.png")
    plt.savefig(plot_path)
