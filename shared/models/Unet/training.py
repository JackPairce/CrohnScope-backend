import os
from typing import List, Literal
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam, lr_scheduler
from torch.types import Device
from app.db.models import Patch
from torchvision import transforms
from app.db.session import SessionLocal
from app.services.ai.Unet.models import UNet
from torch.utils.data import DataLoader, Dataset
from torch import from_numpy, load, save, sigmoid, no_grad
import numpy as np
from tqdm.notebook import tqdm

from app.services.cell.cell_service import count_cells


class PatchDataset(Dataset):
    """
    PyTorch Dataset for loading image and mask patches.
    """

    def __init__(self, patches: List[Patch], transform=None):
        self.patches = patches
        if transform is None:
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
            ])
        self.transform = transform
        
        # Preload and preprocess all patches to memory for faster training
        self.preprocessed_data = []
        for patch in tqdm(patches, desc="Preprocessing patches"):
            img_patch = patch.img_patch
            mask_patch = patch.mask_patch
            
            # Preprocess image
            img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min() + 1e-8)
            img_patch = np.moveaxis(img_patch, -1, 0).astype(np.float32)
            
            # Preprocess mask
            mask_patch = mask_patch.astype(np.float32)
            if len(mask_patch.shape) == 3 and mask_patch.shape[0] == 256:
                class_dim = np.where(np.array(mask_patch.shape) == 2)[0][0]
                mask_patch = np.moveaxis(mask_patch, class_dim, 0)
            elif len(mask_patch.shape) == 2:
                mask_patch = np.expand_dims(mask_patch, axis=0)
                
            # Convert to tensors
            img_tensor = from_numpy(img_patch)
            mask_tensor = from_numpy(mask_patch)
            
            if self.transform:
                img_tensor = self.transform(img_tensor)
                
            self.preprocessed_data.append((img_tensor, mask_tensor))

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx: int):
        return self.preprocessed_data[idx]


# Define loss functions
def dice_coef(pred, target, smooth=1.0):
    """Calculate Dice coefficient"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()

    return (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


def dice_loss(pred, target, smooth=1.0):
    """Dice loss function"""
    return 1 - dice_coef(pred, target, smooth)


def combined_loss(pred, target):
    """Combined BCE and Dice loss with more weight on Dice"""
    bce = BCEWithLogitsLoss()(pred, target)
    pred_sigmoid = sigmoid(pred)
    dice = dice_loss(pred_sigmoid, target)
    return 0.2 * bce + 0.8 * dice  # More emphasis on Dice loss


def Training(
    device: Device,
    dataloader: DataLoader,
    val_loader: DataLoader,
    model_path: str,
    checkpoint_type: Literal["best", "last", None] = None,
    num_epochs=50,
    lr=1e-4,
    early_stopping_patience=10,
) -> tuple[UNet, dict]:
    BEST_WEIGHTS_PATH = f"{model_path}/best_unet_weights.pth"
    LATEST_WEIGHTS_PATH = f"{model_path}/latest_unet_weights.pth"

    session = SessionLocal()
    num_cells = count_cells(session)
    if num_cells < 2:
        raise ValueError("At least two cell types are required for training.")
    session.close()

    model = UNet(in_channels=3, num_classes=num_cells)

    best_loss = float("inf")  # Initialize best loss to infinity
    start_epoch = 0
    
    # Initialize model and optimizer
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr)
    
    # Load checkpoint if specified
    if checkpoint_type:
        if checkpoint_type == "best":
            weights_path = BEST_WEIGHTS_PATH
        elif checkpoint_type == "last":
            weights_path = LATEST_WEIGHTS_PATH
            
        if os.path.exists(weights_path):
            print(f"Loading checkpoint from {weights_path}")
            checkpoint = load(weights_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_loss = checkpoint.get("loss", float("inf"))
            print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
        else:
            print(f"No checkpoint found at {weights_path}, starting from scratch")

    # Initialize optimizer with weight decay
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    no_improvement_count = 0
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_dice_scores = []

    with tqdm(total=num_epochs, desc="Training") as pbar:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = len(dataloader)

            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = combined_loss(outputs, labels)
                epoch_loss += loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(batch_loss=loss.item(), batch=f"{i+1}/{num_batches}")

            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)
            
            # Run validation if validation loader is provided
            if val_loader is not None:
                val_loss, val_dice = validate_model(model, val_loader, device)
                val_losses.append(val_loss)
                val_dice_scores.append(val_dice)
                print(f"\nEpoch {epoch}: Train Loss={avg_epoch_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")
                
                # Use validation loss for model selection and early stopping
                current_loss = val_loss
            else:
                current_loss = avg_epoch_loss
            
            # Save the latest checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": current_loss,
                "train_loss": avg_epoch_loss
            }
            if val_loader is not None:
                checkpoint.update({
                    "val_loss": val_loss,
                    "val_dice": val_dice
                })
            save(checkpoint, LATEST_WEIGHTS_PATH)
            
            # Save the best weights if current loss is lower than previous best
            if current_loss < best_loss:
                best_loss = current_loss
                save(checkpoint, BEST_WEIGHTS_PATH)
                no_improvement_count = 0
                print(f"New best model saved! Best loss: {best_loss:.4f}")
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
                    # Return both the model and the training history
                    history = {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_dice_scores': val_dice_scores
                    }
                    return model, history

            # Update learning rate scheduler
            scheduler.step(avg_epoch_loss)
            
            # Update progress bar for the epoch
            pbar.set_postfix(avg_loss=avg_epoch_loss, best_loss=best_loss)
            pbar.update(1)
        pbar.close()
    
    # Return both the model and the training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_dice_scores': val_dice_scores
    }
    return model, history


def validate_model(model, val_loader, device):
    """Validate the model on validation data"""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = len(val_loader)

    with no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = combined_loss(outputs, labels)
            
            # Apply sigmoid and threshold for binary predictions
            pred = (sigmoid(outputs) > 0.5).float()
            dice = dice_coef(pred, labels)

            total_loss += loss.item()
            total_dice += dice.item()

    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches

    return avg_loss, avg_dice
