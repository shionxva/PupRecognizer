#Operation Verthandi - Stage 2
#Create a trained model

import matplotlib.pyplot as plt # For visualizing images

def plot_metrics(train_losses, val_losses, val_accuracies) -> None:
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
    ax1.plot(epochs, val_losses, label='Val Loss', color='orange')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.plot(epochs, val_accuracies, label='Val Accuracy', color='green')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    plt.title('Training & Validation Loss + Accuracy Over Epochs (Tsinghua)')
    plt.tight_layout()
    plt.show()

train_loss = [
    2.2369, 1.4477, 1.2660, 1.1768, 1.1179,
    1.0775, 1.0311, 1.0186, 0.9808, 0.9580,
    0.9325, 0.9207, 0.8963, 0.8766, 0.8615,
    0.8519, 0.8397, 0.8258, 0.8087, 0.7885,
    0.7859, 0.7729, 0.7763, 0.7535, 0.7538
]

val_loss = [
    1.1577, 0.8817, 0.8208, 0.7690, 0.7507,
    0.7143, 0.7038, 0.6867, 0.6864, 0.6726,
    0.6721, 0.6795, 0.6752, 0.6556, 0.6724,
    0.6577, 0.6594, 0.6743, 0.6613, 0.6666,
    0.6628, 0.6561, 0.6681, 0.6588, 0.6670
]

val_acc = [
    67.91, 73.81, 74.72, 76.55, 76.63,
    77.75, 78.18, 78.52, 78.21, 79.14,
    78.76, 78.64, 78.95, 79.55, 79.13,
    79.42, 79.56, 79.31, 79.54, 79.14,
    79.46, 79.59, 79.22, 79.61, 79.38
]

plot_metrics(train_loss, val_loss, val_acc)