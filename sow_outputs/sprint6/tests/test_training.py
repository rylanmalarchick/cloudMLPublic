"""
Unit tests for training loops and evaluation metrics.

Tests cover:
- Loss computation (MSE, temporal consistency)
- Validation metrics computation
- Early stopping logic
- Learning rate scheduling
- Gradient computation
"""

import numpy as np
import pytest


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    return np.array([1000.0, 1200.0, 1500.0, 800.0, 1100.0])


@pytest.fixture
def sample_targets():
    """Create sample ground truth targets."""
    return np.array([950.0, 1250.0, 1400.0, 850.0, 1050.0])


@pytest.fixture
def sample_temporal_sequence():
    """Create sample temporal sequence of predictions."""
    return np.array(
        [
            [1000.0, 1050.0, 1100.0, 1150.0, 1200.0],  # Sequence 1
            [800.0, 850.0, 900.0, 950.0, 1000.0],  # Sequence 2
        ]
    )


class TestLossComputation:
    """Test loss function computation."""

    def test_mse_loss(self, sample_predictions, sample_targets):
        """Test Mean Squared Error computation."""
        mse = np.mean((sample_predictions - sample_targets) ** 2)

        assert mse >= 0
        assert np.isfinite(mse)
        # Expected MSE for these samples
        expected_mse = np.mean([50**2, 50**2, 100**2, 50**2, 50**2])
        assert np.isclose(mse, expected_mse, atol=1.0)

    def test_mae_loss(self, sample_predictions, sample_targets):
        """Test Mean Absolute Error computation."""
        mae = np.mean(np.abs(sample_predictions - sample_targets))

        assert mae >= 0
        assert np.isfinite(mae)
        # Expected MAE
        expected_mae = np.mean([50, 50, 100, 50, 50])
        assert np.isclose(mae, expected_mae, atol=1.0)

    def test_rmse_loss(self, sample_predictions, sample_targets):
        """Test Root Mean Squared Error computation."""
        mse = np.mean((sample_predictions - sample_targets) ** 2)
        rmse = np.sqrt(mse)

        assert rmse >= 0
        assert np.isfinite(rmse)
        assert rmse >= np.mean(
            np.abs(sample_predictions - sample_targets)
        )  # RMSE >= MAE

    def test_temporal_consistency_loss(self, sample_temporal_sequence):
        """Test temporal consistency loss (smoothness)."""
        # Compute differences between consecutive time steps
        temporal_diffs = np.diff(sample_temporal_sequence, axis=1)

        # L2 norm of differences (penalize large changes)
        consistency_loss = np.mean(temporal_diffs**2)

        assert consistency_loss >= 0
        assert np.isfinite(consistency_loss)

    def test_huber_loss(self, sample_predictions, sample_targets):
        """Test Huber loss (robust to outliers)."""
        delta = 50.0  # Huber delta parameter
        errors = sample_predictions - sample_targets

        # Huber loss: quadratic for small errors, linear for large
        huber = np.where(
            np.abs(errors) <= delta,
            0.5 * errors**2,
            delta * (np.abs(errors) - 0.5 * delta),
        )

        loss = np.mean(huber)

        assert loss >= 0
        assert np.isfinite(loss)


class TestMetricsComputation:
    """Test evaluation metrics computation."""

    def test_r2_score(self, sample_predictions, sample_targets):
        """Test R² score computation."""
        ss_res = np.sum((sample_targets - sample_predictions) ** 2)
        ss_tot = np.sum((sample_targets - np.mean(sample_targets)) ** 2)

        r2 = 1 - (ss_res / ss_tot)

        assert -np.inf < r2 <= 1.0
        assert np.isfinite(r2)

    def test_r2_perfect_prediction(self):
        """Test R² = 1.0 for perfect predictions."""
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = targets.copy()

        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        assert np.isclose(r2, 1.0)

    def test_r2_negative_score(self):
        """Test R² can be negative for poor predictions."""
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([10.0, 10.0, 10.0, 10.0, 10.0])  # Bad predictions

        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        assert r2 < 0  # Worse than predicting mean

    def test_mape(self, sample_predictions, sample_targets):
        """Test Mean Absolute Percentage Error."""
        mape = (
            np.mean(np.abs((sample_targets - sample_predictions) / sample_targets))
            * 100
        )

        assert mape >= 0
        assert np.isfinite(mape)

    def test_correlation(self, sample_predictions, sample_targets):
        """Test Pearson correlation coefficient."""
        correlation = np.corrcoef(sample_predictions, sample_targets)[0, 1]

        assert -1 <= correlation <= 1
        assert np.isfinite(correlation)


class TestEarlyStopping:
    """Test early stopping logic."""

    def test_early_stopping_improvement(self):
        """Test early stopping detects improvement."""
        val_losses = [1.0, 0.9, 0.8, 0.7, 0.6]  # Improving

        patience = 3
        best_loss = float("inf")
        counter = 0

        for loss in val_losses:
            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1

            should_stop = counter >= patience

        assert not should_stop  # Should not stop (always improving)

    def test_early_stopping_no_improvement(self):
        """Test early stopping triggers after patience."""
        val_losses = [1.0, 0.9, 0.91, 0.92, 0.93, 0.94]  # Stops improving

        patience = 3
        best_loss = float("inf")
        counter = 0
        stopped_at_epoch = None

        for epoch, loss in enumerate(val_losses):
            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                stopped_at_epoch = epoch
                break

        assert stopped_at_epoch is not None
        assert (
            stopped_at_epoch == 4
        )  # Should stop at epoch 4 (after 3 epochs of no improvement)

    def test_early_stopping_min_delta(self):
        """Test early stopping with minimum delta threshold."""
        val_losses = [1.0, 0.99, 0.98, 0.97, 0.96]  # Small improvements

        patience = 2
        min_delta = 0.02  # Minimum improvement required
        best_loss = float("inf")
        counter = 0

        for loss in val_losses:
            if loss < best_loss - min_delta:
                best_loss = loss
                counter = 0
            else:
                counter += 1

            should_stop = counter >= patience

        # Should stop because improvements are < min_delta
        assert should_stop


class TestLearningRateScheduling:
    """Test learning rate scheduling."""

    def test_step_decay(self):
        """Test step decay learning rate schedule."""
        initial_lr = 0.001
        decay_factor = 0.1
        decay_epochs = [30, 60, 90]

        current_epoch = 35

        # Compute current learning rate
        lr = initial_lr
        for epoch_threshold in decay_epochs:
            if current_epoch >= epoch_threshold:
                lr *= decay_factor

        assert lr == initial_lr * decay_factor  # One decay step

    def test_exponential_decay(self):
        """Test exponential decay schedule."""
        initial_lr = 0.001
        decay_rate = 0.95
        epoch = 10

        lr = initial_lr * (decay_rate**epoch)

        assert 0 < lr < initial_lr
        assert np.isfinite(lr)

    def test_cosine_annealing(self):
        """Test cosine annealing schedule."""
        initial_lr = 0.001
        min_lr = 0.0001
        max_epochs = 100
        current_epoch = 50

        # Cosine annealing formula
        lr = (
            min_lr
            + (initial_lr - min_lr)
            * (1 + np.cos(np.pi * current_epoch / max_epochs))
            / 2
        )

        assert min_lr <= lr <= initial_lr
        assert np.isfinite(lr)

    def test_warmup_schedule(self):
        """Test learning rate warmup."""
        target_lr = 0.001
        warmup_epochs = 5
        current_epoch = 3

        # Linear warmup
        lr = target_lr * (current_epoch / warmup_epochs)

        assert 0 < lr < target_lr
        assert np.isfinite(lr)


class TestGradientComputation:
    """Test gradient computation and properties."""

    def test_gradient_finite(self):
        """Test gradients are finite."""
        # Simulate gradient
        gradient = np.random.randn(10)

        assert np.all(np.isfinite(gradient))

    def test_gradient_norm(self):
        """Test gradient L2 norm computation."""
        gradient = np.array([1.0, 2.0, 3.0, 4.0])

        grad_norm = np.linalg.norm(gradient)

        expected_norm = np.sqrt(1 + 4 + 9 + 16)
        assert np.isclose(grad_norm, expected_norm)

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        gradient = np.array([10.0, 20.0, 30.0])
        max_norm = 5.0

        grad_norm = np.linalg.norm(gradient)

        if grad_norm > max_norm:
            clipped_gradient = gradient * (max_norm / grad_norm)
        else:
            clipped_gradient = gradient

        clipped_norm = np.linalg.norm(clipped_gradient)

        assert clipped_norm <= max_norm + 1e-6


class TestBatchProcessing:
    """Test batch processing utilities."""

    def test_batch_creation(self):
        """Test creating batches from data."""
        data = np.arange(100)
        batch_size = 10

        n_batches = len(data) // batch_size

        batches = []
        for i in range(n_batches):
            batch = data[i * batch_size : (i + 1) * batch_size]
            batches.append(batch)

        assert len(batches) == 10
        assert all(len(batch) == batch_size for batch in batches)

    def test_batch_remainder(self):
        """Test handling remainder in batching."""
        data = np.arange(105)
        batch_size = 10

        n_full_batches = len(data) // batch_size
        remainder = len(data) % batch_size

        assert n_full_batches == 10
        assert remainder == 5

    def test_batch_shuffling(self):
        """Test batch shuffling."""
        data = np.arange(100)

        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(data))
        shuffled_data = data[shuffled_indices]

        # Shuffled data should have same elements but different order
        assert set(data) == set(shuffled_data)
        assert not np.array_equal(data, shuffled_data)


class TestValidationSplitting:
    """Test train/validation splitting."""

    def test_train_val_split(self):
        """Test train/validation split."""
        data = np.arange(100)
        val_fraction = 0.2

        n_val = int(len(data) * val_fraction)
        n_train = len(data) - n_val

        train_data = data[:n_train]
        val_data = data[n_train:]

        assert len(train_data) == 80
        assert len(val_data) == 20
        assert len(train_data) + len(val_data) == len(data)

    def test_no_data_leakage(self):
        """Test train and validation sets don't overlap."""
        data = np.arange(100)

        train_indices = np.arange(80)
        val_indices = np.arange(80, 100)

        overlap = np.intersect1d(train_indices, val_indices)

        assert len(overlap) == 0


class TestModelCheckpointing:
    """Test model checkpointing logic."""

    def test_save_best_model(self):
        """Test saving best model based on validation loss."""
        val_losses = [1.0, 0.8, 0.9, 0.7, 0.75]

        best_loss = float("inf")
        save_epochs = []

        for epoch, loss in enumerate(val_losses):
            if loss < best_loss:
                best_loss = loss
                save_epochs.append(epoch)

        # Should save at epochs 0, 1, 3
        assert save_epochs == [0, 1, 3]

    def test_checkpoint_frequency(self):
        """Test checkpointing at regular intervals."""
        total_epochs = 100
        checkpoint_interval = 10

        checkpoint_epochs = []
        for epoch in range(total_epochs):
            if epoch % checkpoint_interval == 0:
                checkpoint_epochs.append(epoch)

        assert len(checkpoint_epochs) == 10
        assert checkpoint_epochs == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


class TestDataAugmentation:
    """Test data augmentation techniques."""

    def test_random_flip(self):
        """Test random horizontal flip."""
        image = np.random.rand(224, 224, 3)

        # Flip horizontally
        flipped = np.fliplr(image)

        assert flipped.shape == image.shape
        assert not np.array_equal(image, flipped)

    def test_random_crop(self):
        """Test random crop."""
        image = np.random.rand(256, 256, 3)
        crop_size = 224

        # Random crop
        y = np.random.randint(0, 256 - crop_size)
        x = np.random.randint(0, 256 - crop_size)

        cropped = image[y : y + crop_size, x : x + crop_size]

        assert cropped.shape == (224, 224, 3)

    def test_normalization(self):
        """Test image normalization."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0

        assert 0 <= normalized.min() <= normalized.max() <= 1
        assert normalized.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
