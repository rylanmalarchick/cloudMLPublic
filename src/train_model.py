# in src/train_model.py

import os
import time
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import r2_score

# --- CORRECTED IMPORT ---
from .main_utils import get_device, save_model_and_scaler
from .pytorchmodel import CustomLoss


class WarmupAndReduceLROnPlateau(_LRScheduler):
    """
    LR scheduler that warms up for a specified number of steps,
    then switches to ReduceLROnPlateau.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        patience=5,
        factor=0.5,
        min_lr=1e-6,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=patience, factor=factor, min_lr=min_lr
        )
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warmup_lrs = [
            np.linspace(0, base_lr, warmup_steps) for base_lr in self.base_lrs
        ]
        self.step_count = 0
        super().__init__(optimizer, -1)

    def get_lr(self):
        if self.step_count < self.warmup_steps:
            return [lrs[self.step_count] for lrs in self.warmup_lrs]
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, metrics=None):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.warmup_lrs[i][self.step_count]
        elif metrics is not None:
            self.reduce_on_plateau.step(metrics)
        else:
            pass


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    scaler,  # Add scaler argument
    hard_mining_k=0.25,
    hard_mining_factor=2.0,
    grad_clip_value=None,
    variance_lambda=0.5,
):
    """Trains the model for one epoch with curriculum learning and variance preservation."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Train", leave=False, dynamic_ncols=True)

    for batch in progress_bar:
        img_stack, sza, saa, y_true, _, _ = [b.to(device) for b in batch]
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):  # Mixed precision
            y_pred, _ = model(img_stack, sza, saa)
            individual_losses = criterion(y_pred, y_true, reduction="none")

            num_hard_samples = int(len(individual_losses) * hard_mining_k)
            if num_hard_samples > 0:
                hard_indices = torch.topk(individual_losses, k=num_hard_samples).indices
                weights = torch.ones_like(individual_losses)
                weights[hard_indices] *= hard_mining_factor
                base_loss = (individual_losses * weights).mean()
            else:
                base_loss = individual_losses.mean()

            # Add variance-preserving term to prevent collapse (only if batch has >1 sample)
            if y_pred.numel() > 1:
                pred_var = y_pred.var()
                target_var = y_true.var()
                variance_loss = (1.0 - pred_var / (target_var + 1e-8)) ** 2
                loss = base_loss + variance_lambda * variance_loss
            else:
                # Single sample batch - skip variance term
                loss = base_loss

        scaler.scale(loss).backward()  # Scale loss and backpropagate

        scaler.unscale_(optimizer)
        if grad_clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, variance_lambda=0.5, y_scaler=None):
    """Validates the model on the validation set with variance-preserving loss and R² calculation.

    Args:
        y_scaler: StandardScaler for unscaling predictions to compute R² in original units (km)
    """
    model.eval()
    total_loss = 0
    all_preds_scaled = []
    all_targets_scaled = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", leave=False, dynamic_ncols=True):
            img_stack, sza, saa, y_true, _, _ = [b.to(device) for b in batch]
            y_true = y_true.squeeze()  # Squeeze y_true to match y_pred's shape

            with torch.amp.autocast("cuda"):  # Mixed precision
                y_pred, _ = model(img_stack, sza, saa)
                y_pred = y_pred.squeeze()

                # Base loss
                base_loss = criterion(y_pred, y_true, reduction="mean")

                # Variance-preserving term (only if batch has >1 sample)
                if y_pred.dim() > 0 and y_pred.numel() > 1:
                    pred_var = y_pred.var()
                    target_var = y_true.var()
                    variance_loss = (1.0 - pred_var / (target_var + 1e-8)) ** 2
                    loss = base_loss + variance_lambda * variance_loss
                else:
                    # Single sample batch - skip variance term
                    loss = base_loss

            total_loss += loss.item()

            # Handle both scalar and vector predictions
            if y_pred.dim() == 0:
                all_preds_scaled.append(y_pred.cpu().numpy().item())
                all_targets_scaled.append(y_true.cpu().numpy().item())
            else:
                all_preds_scaled.extend(y_pred.cpu().numpy().tolist())
                all_targets_scaled.extend(y_true.cpu().numpy().tolist())

    # Convert to numpy arrays
    all_preds_scaled = np.array(all_preds_scaled)
    all_targets_scaled = np.array(all_targets_scaled)

    # Unscale predictions and targets for R² calculation (if scaler provided)
    if y_scaler is not None:
        all_preds_unscaled = y_scaler.inverse_transform(
            all_preds_scaled.reshape(-1, 1)
        ).flatten()
        all_targets_unscaled = y_scaler.inverse_transform(
            all_targets_scaled.reshape(-1, 1)
        ).flatten()
        r2 = r2_score(all_targets_unscaled, all_preds_unscaled)

        # Debug logging
        print(
            f"  DEBUG: Scaled pred range: [{all_preds_scaled.min():.3f}, {all_preds_scaled.max():.3f}]"
        )
        print(
            f"  DEBUG: Unscaled pred range: [{all_preds_unscaled.min():.3f}, {all_preds_unscaled.max():.3f}] km"
        )
        print(
            f"  DEBUG: Unscaled target range: [{all_targets_unscaled.min():.3f}, {all_targets_unscaled.max():.3f}] km"
        )
    else:
        # Fallback: calculate R² on scaled values (not recommended)
        r2 = r2_score(all_targets_scaled, all_preds_scaled)
        print("  WARNING: No y_scaler provided, calculating R² on scaled values")

    # Calculate variance ratio (on scaled values, as this is what the loss uses)
    pred_std = np.std(all_preds_scaled)
    target_std = np.std(all_targets_scaled)
    variance_ratio = pred_std / (target_std + 1e-8)

    return (
        total_loss / len(val_loader),
        r2,
        variance_ratio,
        pred_std,
        all_preds_scaled.max() - all_preds_scaled.min(),
    )


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    config,
    save_path,
    log_dir,
    scaler=None,
):
    """Main training loop with early stopping."""
    device = get_device()
    # Extract base log_dir if it ends with 'tensorboard'
    base_log_dir = log_dir.replace("/tensorboard", "").replace("\\tensorboard", "")
    log_file = os.path.join(base_log_dir, "csv", f"{config['save_name']}.csv")

    # Ensure the logs/csv directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    writer = SummaryWriter(os.path.join(log_dir, config["save_name"]))
    criterion = CustomLoss(
        loss_type=config.get("loss_type", "huber"),
        alpha=config.get("loss_alpha", 0.5),
        huber_delta=config.get("huber_delta", 1.0),
    ).to(device)

    scheduler = WarmupAndReduceLROnPlateau(
        optimizer,
        warmup_steps=config.get("warmup_steps", 500),
        patience=config["early_stopping_patience"],
        factor=0.5,
    )

    best_val_loss = float("inf")
    best_r2 = float("-inf")
    patience_counter = 0
    log_data = []

    # Get variance lambda from config
    variance_lambda = config.get("variance_lambda", 0.5)

    # Initialize GradScaler for mixed precision training
    grad_scaler = torch.amp.GradScaler("cuda")

    print(f"Starting training for {config['epochs']} epochs on {device.upper()}")
    print(
        f"Early stopping enabled. Patience: {config['early_stopping_patience']}, min_delta: {config['early_stopping_min_delta']}"
    )
    print(f"Variance-preserving loss enabled with lambda={variance_lambda}")
    print("Early stopping will track R² score (minimize -R²)")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start_time = time.time()
        avg_train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_scaler,  # Pass grad_scaler
            hard_mining_k=config.get("hard_mining_k", 0.25),
            hard_mining_factor=config.get("hard_mining_factor", 2.0),
            grad_clip_value=config.get("grad_clip_value", None),
            variance_lambda=variance_lambda,
        )
        # Extract y_scaler from scaler dict if provided
        y_scaler = scaler.get("y") if scaler and isinstance(scaler, dict) else None

        avg_val_loss, val_r2, variance_ratio, pred_std, pred_range = validate(
            model,
            val_loader,
            criterion,
            device,
            variance_lambda=variance_lambda,
            y_scaler=y_scaler,
        )
        epoch_duration = time.time() - epoch_start_time

        print(
            f"Epoch {epoch}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"R²: {val_r2:.4f} | Var Ratio: {variance_ratio:.1%} | Pred Std: {pred_std:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_duration:.2f}s"
        )

        scheduler.step(avg_val_loss)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/R2", val_r2, epoch)
        writer.add_scalar("Metrics/VarianceRatio", variance_ratio, epoch)
        writer.add_scalar("Metrics/PredStd", pred_std, epoch)
        writer.add_scalar("Metrics/PredRange", pred_range, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)
        log_data.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "r2": val_r2,
                "variance_ratio": variance_ratio,
                "pred_std": pred_std,
                "pred_range": pred_range,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Early stopping based on R² with minimum variance ratio requirement
        # RUN 5 FIX: Only save model if variance ratio is acceptable (>35%)
        min_variance_ratio = config.get("min_variance_ratio", 0.35)
        if val_r2 > best_r2 + config["early_stopping_min_delta"]:
            if variance_ratio >= min_variance_ratio:
                best_r2 = val_r2
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(
                    f"New best model saved! (R²: {best_r2:.4f}, var_ratio: {variance_ratio:.1%}, val_loss: {best_val_loss:.4f})"
                )
                save_model_and_scaler(model, scaler, save_path)
            else:
                patience_counter += 1
                print(
                    f"R² improved but variance too low ({variance_ratio:.1%} < {min_variance_ratio:.1%}), not saving. "
                    f"patience_counter={patience_counter}/{config['early_stopping_patience']}"
                )
        else:
            patience_counter += 1
            print(
                f"No improvement in R², patience_counter={patience_counter}/{config['early_stopping_patience']}"
            )

        # Emergency stop if severe variance collapse detected
        if variance_ratio < 0.05:
            print(
                f"WARNING: Severe variance collapse detected (ratio={variance_ratio:.3f}). Stopping early."
            )
            break

        if patience_counter >= config["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    writer.close()
    pd.DataFrame(log_data).to_csv(log_file, index=False)
    print(f"Training logs saved to {log_file}")

    # Return the best saved model, not the last one
    model.load_state_dict(torch.load(save_path, weights_only=False)["model_state_dict"])
    return model, log_file
