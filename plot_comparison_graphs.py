import os
import numpy as np
import matplotlib.pyplot as plt


def parse_summary_file(path):
    """
    Expects file format like:
      Pixel_Accuracy: 0.9699
      Mean_IoU: 0.8891
      Per_class_IoU: 0.8389, 0.9551, 0.8733
    """
    pixel_acc = None
    mean_iou = None
    per_class_iou = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Pixel_Accuracy"):
                # e.g. "Pixel_Accuracy: 0.9699"
                pixel_acc = float(line.split(":")[1])
            elif line.startswith("Mean_IoU"):
                mean_iou = float(line.split(":")[1])
            elif line.startswith("Per_class_IoU"):
                # right side: "0.8389, 0.9551, 0.8733"
                right = line.split(":", 1)[1].strip()
                parts = [p.strip() for p in right.split(",")]
                per_class_iou = [float(p) for p in parts if p]

    if pixel_acc is None or mean_iou is None or per_class_iou is None:
        raise ValueError(f"Could not parse all fields from {path}")

    return pixel_acc, mean_iou, per_class_iou


def main():

    paths = {
        "DeepLabV3": "eval_results_deeplabv3/summary_val_deeplabv3.txt",
        "DeepLabV3+ Official": "eval_results_deeplabv3plus/summary_val.txt",
        "Enhanced V3+ (Ours)": "eval_results_enhanced_v3/summary_val.txt",
    }

    class_names = ["Obstacle (0)", "Water (1)", "Sky (2)"]

    model_names = []
    pixel_accs = []
    mean_ious = []
    per_class_ious_all = []


    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Summary file not found: {path}")

        pa, miou, per_class = parse_summary_file(path)
        model_names.append(name)
        pixel_accs.append(pa)
        mean_ious.append(miou)
        per_class_ious_all.append(per_class)

        print(f"{name}:")
        print(f"  Pixel Acc = {pa:.4f}")
        print(f"  Mean IoU  = {miou:.4f}")
        print(f"  Per-class IoU = {[f'{x:.4f}' for x in per_class]}")
        print()

    pixel_accs = np.array(pixel_accs)
    mean_ious = np.array(mean_ious)
    per_class_ious_all = np.array(per_class_ious_all)  # shape: (num_models, num_classes)


    x = np.arange(len(model_names))

    # plt.figure(figsize=(8, 5))
    # plt.bar(x, mean_ious)
    # plt.xticks(x, model_names, rotation=15, ha="right")
    # plt.ylabel("Mean IoU")
    # # plt.title("Mean IoU Comparison (Val)")
    # plt.tight_layout()
    # os.makedirs("graphs", exist_ok=True)
    # plt.savefig("graphs/miou_comparison.png", dpi=300)
    # plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x, mean_ious, color='#004285', alpha=0.7, label='Pixel Acc')
    plt.plot(x, mean_ious, marker='o', color='#C66218', linewidth=2, label='Trend', zorder=3)
    for i, acc in enumerate(mean_ious):
        plt.text(x[i], acc + 0.001, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.ylim(0.8, 1.0)  
    plt.ylabel("Mean IoU")
    plt.tight_layout()
    plt.savefig("graphs/miou_comparison.png", dpi=300)
    plt.close()

   
    # plt.figure(figsize=(8, 5))
    # plt.bar(x, pixel_accs)
    # plt.xticks(x, model_names, rotation=15, ha="right")
    # # plt.plot(x, model_names, marker='o', color='blue', linewidth=2, label='Trend')
    # plt.ylim(0.95, 1.0)
    # plt.ylabel("Pixel Accuracy")
    # # plt.title("Pixel Accuracy Comparison (Val)")
    # plt.tight_layout()
    # plt.savefig("graphs/pixel_accuracy_comparison.png", dpi=300)
    # plt.close()


    plt.figure(figsize=(8, 5))

    plt.bar(x, pixel_accs, color='#004285', alpha=0.7, label='Pixel Acc')


    plt.plot(x, pixel_accs, marker='o', color='#C66218', linewidth=2, label='Trend', zorder=3)

    for i, acc in enumerate(pixel_accs):
        plt.text(x[i], acc + 0.001, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.ylim(0.95, 1.0)  
    plt.ylabel("Pixel Accuracy")
    # plt.title("Pixel Accuracy Comparison (Val)")

    # Handle Legend - Place it outside if it crowds the bars
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig("graphs/pixel_accuracy_comparison.png", dpi=300)
    plt.close()


    num_models = len(model_names)
    num_classes = per_class_ious_all.shape[1]


    indices = np.arange(num_classes)
    bar_width = 0.25

    plt.figure(figsize=(9, 5))

    for i in range(num_models):
        offset = (i - (num_models - 1) / 2) * bar_width
        plt.bar(indices + offset, per_class_ious_all[i], width=bar_width, label=model_names[i])

    plt.xticks(indices, class_names)
    plt.ylim(0.7, 1.0)
    plt.ylabel("IoU")
    # plt.title("Per-class IoU Comparison (Val)")
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig("graphs/per_class_iou_comparison.png", dpi=300)
    plt.close()

    print("Saved graphs in the 'graphs' folder:")
    print("  - graphs/miou_comparison.png")
    print("  - graphs/pixel_accuracy_comparison.png")
    print("  - graphs/per_class_iou_comparison.png")


if __name__ == "__main__":
    main()
