import os
import re
import torch
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


def load_metrics_from_checkpoints(ckpt_dir="checkpoints"):
    """
    Load epoch, val_loss, val_acc, val_miou from all checkpoint files.
    Returns lists sorted by epoch.
    """
    epochs = []
    val_losses = []
    val_accs = []
    val_mious = []

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]

    if not files:
        raise RuntimeError(f"No .pth files found in {ckpt_dir}")

    for fname in files:
        path = os.path.join(ckpt_dir, fname)
        ckpt = torch.load(path, map_location="cpu")

        if not isinstance(ckpt, dict) or "epoch" not in ckpt:
            continue

        epoch = ckpt.get("epoch", None)
        val_loss = ckpt.get("val_loss", None)
        val_acc = ckpt.get("val_acc", None)
        val_miou = ckpt.get("val_miou", None)

        if epoch is None:
            m = re.search(r"epoch_(\d+)", fname)
            if m:
                epoch = int(m.group(1))
            else:
                continue

        epochs.append(epoch)
        val_losses.append(float(val_loss) if val_loss is not None else float("nan"))
        val_accs.append(float(val_acc) if val_acc is not None else float("nan"))
        val_mious.append(float(val_miou) if val_miou is not None else float("nan"))

    # Sort everything by epoch
    sorted_indices = sorted(range(len(epochs)), key=lambda i: epochs[i])
    epochs = [epochs[i] for i in sorted_indices]
    val_losses = [val_losses[i] for i in sorted_indices]
    val_accs = [val_accs[i] for i in sorted_indices]
    val_mious = [val_mious[i] for i in sorted_indices]

    return epochs, val_losses, val_accs, val_mious


def plot_curves(epochs, val_losses, val_accs, val_mious, out_dir="plots"):
    """
    Create and save training curves:
    - val_loss vs epoch
    - val_acc vs epoch
    - val_mIoU vs epoch
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Validation loss
    plt.figure()
    plt.plot(epochs, val_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_loss_curve.png"))
    plt.close()

    # 2) Validation accuracy
    plt.figure()
    plt.plot(epochs, val_accs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Pixel Accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_accuracy_curve.png"))
    plt.close()

    # 3) Validation mIoU
    plt.figure()
    plt.plot(epochs, val_mious, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation mIoU")
    plt.title("Validation mIoU vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_miou_curve.png"))
    plt.close()

    print(f"Saved curves to '{out_dir}/' directory.")


def main():
    ckpt_dir = "checkpoints_enhanced_v3"   
    out_dir = "plots_enhanced_v3"

    epochs, val_losses, val_accs, val_mious = load_metrics_from_checkpoints(ckpt_dir)
    print("Loaded metrics from checkpoints:")
    print("  Epochs:", epochs[0], "...", epochs[-1])
    print("  First 3 val losses:", val_losses[:3])
    print("  First 3 val mIoUs:", val_mious[:3])

    plot_curves(epochs, val_losses, val_accs, val_mious, out_dir=out_dir)


if __name__ == "__main__":
    main()
