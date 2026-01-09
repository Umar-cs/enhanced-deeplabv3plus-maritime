import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
from tqdm import tqdm

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from update_dataloader_torch import LaRSDataset
from enhanced_deeplabv3plus_v3 import DeepLabV3PlusV3
from utils import calculate_pixel_accuracy, calculate_iou



def compute_horizon_targets(masks, num_bins=32, sky_class=2, ignore_index=255):
    """
    Compute horizon bin targets from semantic masks.

    Classes (LaRS):
      0: obstacle
      1: water
      2: sky
    We treat rows where sky > 50% as 'above horizon'.
    Horizon = last sky-dominant row.
    """
    B, H, W = masks.shape
    device = masks.device
    targets = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        mask_b = masks[b]

        valid = (mask_b != ignore_index)
        sky_mask = (mask_b == sky_class) & valid

        sky_counts = sky_mask.sum(dim=1).float()
        valid_counts = valid.sum(dim=1).float().clamp(min=1.0)
        sky_ratio = sky_counts / valid_counts

        sky_rows = (sky_ratio > 0.5).nonzero(as_tuple=False).view(-1)

        if len(sky_rows) == 0:
            horizon_row = int(0.3 * (H - 1))
        else:
            horizon_row = sky_rows.max().item()

        bin_idx = int(horizon_row / max(H - 1, 1) * (num_bins - 1))
        bin_idx = max(0, min(num_bins - 1, bin_idx))
        targets[b] = bin_idx

    return targets



def save_predictions(preds, masks, epoch, out_dir="predictions_enhanced_v3_new", num_classes=3):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(preds)):
        p = torch.argmax(preds[i], dim=0).cpu().numpy().astype(np.uint8)
        m = masks[i].cpu().numpy().astype(np.uint8)

        scale = 255 // max(1, (num_classes - 1))
        pred_img = (p * scale).astype(np.uint8)
        mask_img = (m * scale).astype(np.uint8)

        Image.fromarray(pred_img).save(os.path.join(out_dir, f"epoch{epoch}_pred_{i}.png"))
        Image.fromarray(mask_img).save(os.path.join(out_dir, f"epoch{epoch}_mask_{i}.png"))


def main():
    img_size = (512, 512)
    batch_size = 16
    num_classes = 3

    BASE_EPOCHS = 50         
    MAX_EXTRA_EPOCHS = 10    
    TOTAL_MAX_EPOCHS = BASE_EPOCHS + MAX_EXTRA_EPOCHS

    lr = 1e-3

    HORIZON_BINS = 32
    AUX_WEIGHT = 0.4
    HORIZON_WEIGHT = 0.02    
    MAX_GRAD_NORM = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Enhanced V3 on device:", device)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
    ])

    IMAGE_ROOT = "LaRS_data/images"
    ANN_ROOT = "LaRS_data/annotations"

    # ---- Datasets ----
    train_set = LaRSDataset(
        image_list_path=os.path.join(IMAGE_ROOT, "train", "image_list.txt").replace("\\", "/"),
        image_dir=os.path.join(IMAGE_ROOT, "train", "images").replace("\\", "/"),
        mask_dir=os.path.join(ANN_ROOT, "train", "semantic_masks").replace("\\", "/"),
        annotation_json_path=os.path.join(ANN_ROOT, "train", "image_annotations.json").replace("\\", "/"),
        img_size=img_size,
        transform=train_transform,
        debug=False,
    )

    val_set = LaRSDataset(
        image_list_path=os.path.join(IMAGE_ROOT, "val", "image_list.txt").replace("\\", "/"),
        image_dir=os.path.join(IMAGE_ROOT, "val", "images").replace("\\", "/"),
        mask_dir=os.path.join(ANN_ROOT, "val", "semantic_masks").replace("\\", "/"),
        annotation_json_path=os.path.join(ANN_ROOT, "val", "image_annotations.json").replace("\\", "/"),
        img_size=img_size,
        transform=None,
        debug=False,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    print("Train samples:", len(train_set), "| Val samples:", len(val_set))

    model = DeepLabV3PlusV3(num_classes=num_classes, horizon_bins=HORIZON_BINS, pretrained=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_MAX_EPOCHS)

    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    horizon_criterion = nn.CrossEntropyLoss()

    best_val_iou = 0.0
    best_epoch = 0
    val_miou_history = []

    os.makedirs("checkpoints_enhanced_v3_new", exist_ok=True)


    for epoch in range(1, TOTAL_MAX_EPOCHS + 1):
        print(f"\n=== [Enhanced V3] Epoch {epoch}/{TOTAL_MAX_EPOCHS} ===")
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, masks, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)

            main_logits = outputs["main"]
            aux_logits = outputs["aux"]
            horizon_logits = outputs["horizon"]

            loss_main = seg_criterion(main_logits, masks)
            loss_aux = seg_criterion(aux_logits, masks)

            horizon_targets = compute_horizon_targets(masks, num_bins=HORIZON_BINS)
            loss_horizon = horizon_criterion(horizon_logits, horizon_targets)

            loss = loss_main + AUX_WEIGHT * loss_aux + HORIZON_WEIGHT * loss_horizon

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()

            acc = calculate_pixel_accuracy(outputs, masks)
            running_loss += loss.item()
            running_acc += acc

            if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
                print(f"[Train] Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        print(f"[Train] Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        all_ious = []

        with torch.no_grad():
            for idx, (images, masks, _) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                outputs = model(images)
                main_logits = outputs["main"]
                aux_logits = outputs["aux"]
                horizon_logits = outputs["horizon"]

                loss_main = seg_criterion(main_logits, masks)
                loss_aux = seg_criterion(aux_logits, masks)
                horizon_targets = compute_horizon_targets(masks, num_bins=HORIZON_BINS)
                loss_horizon = horizon_criterion(horizon_logits, horizon_targets)

                loss = loss_main + AUX_WEIGHT * loss_aux + HORIZON_WEIGHT * loss_horizon

                acc = calculate_pixel_accuracy(outputs, masks)
                ious = calculate_iou(outputs, masks, num_classes=num_classes)

                val_loss += loss.item()
                val_acc += acc
                all_ious.append(ious)

                if idx < 3:
                    save_predictions(main_logits.cpu(), masks.cpu(), epoch)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        mean_ious = np.nanmean(np.array(all_ious), axis=0)
        mean_iou = float(np.nanmean(mean_ious))

        val_miou_history.append(mean_iou)

        print(f"[Val]  Epoch {epoch} | Loss: {avg_val_loss:.4f} "
              f"| Acc: {avg_val_acc:.4f} | mIoU: {mean_iou:.4f}")

        # ---- Save checkpoint ----
        ckpt_path = os.path.join(
            "checkpoints_enhanced_v3_new",
            f"epoch_{epoch}_miou_{mean_iou:.4f}.pth"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "val_miou": mean_iou,
        }, ckpt_path)

        # ----best model ----
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            best_epoch = epoch
            torch.save(model.state_dict(), "checkpoints_enhanced_v3_new/best_model_v3.pth")
            print(f"[INFO] New BEST Enhanced V3 model (mIoU={best_val_iou:.4f}) at epoch {epoch}")


        if epoch == BASE_EPOCHS:
            print("\n=== Reached base 50 epochs ===")
            print(f"Best mIoU so far: {best_val_iou:.4f} at epoch {best_epoch}")
            if best_epoch >= BASE_EPOCHS - 2:
                print("Validation mIoU is still improving near epoch 50.")
                print("-> Continuing training up to 60 epochs (if improvement continues).")
            else:
                print("Validation mIoU has plateaued before epoch 50.")
                print("-> Stopping training at 50 epochs.")
                break

    print("\nTraining finished.")
    print(f"Best validation mIoU: {best_val_iou:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
