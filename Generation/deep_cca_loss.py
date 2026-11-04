"""Training-only Deep CCA auxiliary loss for paired EEG/CLIP embeddings.

This module does NOT change the 1024-dimensional ATMS encoder output used at
inference. Small modality-specific networks are optimized only during training.

See: Andrew et al., "Deep Canonical Correlation Analysis", ICML 2013.
"""

import torch
from torch import nn


def regularized_cca_loss(eeg_view, image_view, regularization=1e-3):
    """Return 1 - mean regularized canonical correlation.

    eeg_view, image_view: matching [batch, cca_dim] tensors. cca_dim must be
    smaller than batch size to avoid unregularized high-dimensional CCA.
    Full precision is used for the small covariance/Cholesky/SVD operations.
    """
    if eeg_view.ndim != 2 or image_view.ndim != 2:
        raise ValueError("CCA inputs must have shape [batch, cca_dim]")
    if eeg_view.shape != image_view.shape:
        raise ValueError("Paired CCA views must have identical shapes")

    batch_size, cca_dim = eeg_view.shape
    if batch_size <= cca_dim:
        raise ValueError(
            f"CCA requires batch_size > cca_dim, got {batch_size} <= {cca_dim}"
        )
    if regularization <= 0:
        raise ValueError("regularization must be positive")

    x = eeg_view.float()
    y = image_view.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    inv_n = 1.0 / (batch_size - 1)

    identity = torch.eye(cca_dim, device=x.device, dtype=x.dtype)
    cov_x = x.T @ x * inv_n + regularization * identity
    cov_y = y.T @ y * inv_n + regularization * identity
    cov_xy = x.T @ y * inv_n

    # Cholesky whitening avoids eigendecomposition eigenvector gradients.
    lx = torch.linalg.cholesky(cov_x)
    ly = torch.linalg.cholesky(cov_y)
    whitened = torch.linalg.solve_triangular(lx, cov_xy, upper=False)
    whitened = torch.linalg.solve_triangular(
        ly, whitened.T, upper=False
    ).T
    correlations = torch.linalg.svdvals(whitened)
    return 1.0 - correlations.mean()


class DeepCCALoss(nn.Module):
    """Two trainable view-specific heads with a regularized CCA objective.

    The ATMS encoder and its 1024-D CLIP prediction path remain unchanged.
    This module and its parameters are needed during training only.
    """

    def __init__(self, input_dim=1024, hidden_dim=256, cca_dim=16,
                 regularization=1e-3):
        super().__init__()
        if not (0 < cca_dim < 64):
            raise ValueError("cca_dim must be between 1 and 63 for batch_size=64")
        self.regularization = regularization
        self.eeg_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cca_dim),
        )
        self.clip_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cca_dim),
        )

    def forward(self, predicted_clip, true_clip):
        if predicted_clip.shape != true_clip.shape:
            raise ValueError("Predicted and true CLIP features must have same shape")
        x = self.eeg_head(predicted_clip.float())
        y = self.clip_head(true_clip.detach().float())
        return regularized_cca_loss(x, y, self.regularization)
