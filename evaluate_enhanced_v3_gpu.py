import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from update_dataloader_torch import LaRSDataset
from enhanced_deeplabv3plus_v3 import DeepLabV3PlusV3
from utils import calculate_pixel_accuracy, calculate_iou


def evaluate_enhanced_v3(
    checkpoint_path="checkpoints_enhanced_v3/best_model_v3.pth",
    image_root="./LaRS_data/images",
    annotation_root="./LaRS_data/annotations",
    split="val",
    num_classes=3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n======================================")
    print(f" Evaluating Enhanced V3 model on split '{split}'")
    print("======================================")
    print("Using device:", device)

    masks_dir = os.path.join(annotation_root, split, "semantic_masks")
    has_masks = os.path.isdir(masks_dir)
    if has_masks:
        print(f"Found semantic masks at: {masks_dir}")
    else:
        print(f"No semantic masks for split '{split}'. Inference only.")
    mask_dir = masks_dir if has_masks else None

    json_path = os.path.join(annotation_root, split, "image_annotations.json").replace("\\", "/")

    dataset = LaRSDataset(
        image_list_path=os.path.join(image_root, split, "image_list.txt").replace("\\", "/"),
        image_dir=os.path.join(image_root, split, "images").replace("\\", "/"),
        mask_dir=mask_dir,
        annotation_json_path=json_path,
        img_size=(512, 512),
        transform=None,
        debug=False,
    )

    if len(dataset) == 0:
        print("Dataset is empty. Check paths.")
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Total samples in {split}: {len(dataset)}")

    # ----- load model -----
    model = DeepLabV3PlusV3(num_classes=num_classes, horizon_bins=32, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # ----- accumulators -----
    total_acc = 0.0
    total_samples = 0
    all_ious = []
    all_preds_flat = []
    all_masks_flat = []

    pred_dir = f"predictions_enhanced_v3_{split}"
    os.makedirs(pred_dir, exist_ok=True)

    print("Running evaluation...")
    with torch.no_grad():
        for idx, (images, masks, _) in enumerate(tqdm(loader)):
            images = images.to(device)
            if has_masks:
                masks = masks.to(device)

            outputs = model(images)
            logits = outputs["main"] 

            # save prediction image
            preds = torch.argmax(logits, dim=1)
            pred_np = preds[0].cpu().numpy().astype(np.uint8)
            scale = 255 // max(1, (num_classes - 1))
            Image.fromarray(pred_np * scale).save(
                os.path.join(pred_dir, f"pred_{idx:05d}.png")
            )

            if has_masks and masks.numel() > 0:
                metric_output = {"main": logits}
                acc = calculate_pixel_accuracy(metric_output, masks)
                ious = calculate_iou(metric_output, masks, num_classes=num_classes)

                total_acc += acc
                total_samples += 1
                all_ious.append(ious)

                preds_flat = preds.view(-1).cpu().numpy()
                masks_flat = masks.view(-1).cpu().numpy()
                valid = masks_flat != 255

                all_preds_flat.extend(preds_flat[valid].tolist())
                all_masks_flat.extend(masks_flat[valid].tolist())

    if has_masks and total_samples > 0:
        mean_acc = total_acc / total_samples
        all_ious_np = np.array(all_ious)
        per_class_iou = np.nanmean(all_ious_np, axis=0)
        miou = float(np.nanmean(per_class_iou))

        cm = confusion_matrix(all_masks_flat, all_preds_flat, labels=list(range(num_classes)))

        print("\n=========== RESULTS (Enhanced V3) ===========")
        print(f"Pixel Accuracy: {mean_acc:.4f}")
        print(f"Mean IoU:       {miou:.4f}")
        print("Per-class IoU:")
        for cls_id, iou in enumerate(per_class_iou):
            print(f"  Class {cls_id}: {iou:.4f}")
        print("\nConfusion Matrix:")
        print(cm)

        os.makedirs("eval_results_enhanced_v3", exist_ok=True)
        np.save(f"eval_results_enhanced_v3/confusion_matrix_{split}.npy", cm)
        np.save(f"eval_results_enhanced_v3/per_class_iou_{split}.npy", per_class_iou)
        with open(f"eval_results_enhanced_v3/summary_{split}.txt", "w") as f:
            f.write(f"Pixel_Accuracy: {mean_acc:.4f}\n")
            f.write(f"Mean_IoU: {miou:.4f}\n")
            f.write("Per_class_IoU: " + ", ".join(f"{x:.4f}" for x in per_class_iou) + "\n")

        print(f"\nSaved summary to eval_results_enhanced_v3/summary_{split}.txt")
    else:
        print("\nNo masks for this split. Saved predictions to:", pred_dir)


if __name__ == "__main__":
    evaluate_enhanced_v3(
        checkpoint_path="checkpoints_enhanced_v3/best_model_v3.pth",
        image_root="LaRS_data/images",
        annotation_root="LaRS_data/annotations",
        split="val",     
        num_classes=3,
    )
