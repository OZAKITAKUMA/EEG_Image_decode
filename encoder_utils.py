"""
Shared ATMS encoder training utilities.

Used by both Generation (train.py) and Retrieval (train.py).

Two loss modes controlled by ``loss_mode``:

  'generation'
      Combined MSE regression + CLIP contrastive loss on raw (unnormalized) features.
      Encourages the EEG embedding to sit close to the corresponding image embedding
      in CLIP space, suitable as a prior for diffusion-based reconstruction.

          loss = alpha * MSELoss(eeg, img) * 10 + (1-alpha) * CLIPLoss(eeg, img) * 10

  'retrieval'
      Pure CLIP contrastive loss on L2-normalized features, with an optional
      text-supervision term.  Optimises for ranked image retrieval accuracy.

          eeg_n, img_n, txt_n = L2_norm(eeg), L2_norm(img), L2_norm(txt)
          loss = alpha * CLIPLoss(eeg_n, img_n) + (1-alpha) * CLIPLoss(eeg_n, txt_n)

Public API
----------
  train_encoder_epoch(sub, model, loader, optimizer, device,
                      img_features_all, *,
                      loss_mode, alpha, text_features_all=None)
        → (avg_loss, accuracy)

  evaluate_encoder(sub, model, loader, device, img_features_all, *,
                   k=200, loss_mode, alpha, text_features_all=None)
        → (avg_loss, k_way_accuracy)

  stratified_condition_split(n_classes, conditions_per_class,
                             trials_per_condition, val_ratio, seed)
        → (train_indices, val_indices)
"""

import random
import re

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_subject_id(sub: str) -> int:
    match = re.search(r'\d+$', sub)
    return int(match.group()) if match else 0

def _make_subject_ids(
    sub,
    model,
    batch_size,
    device,
    use_subject_id,
):
    if use_subject_id:
        subject_id = _extract_subject_id(sub)
    else:
        embedding_layer = (
            model.encoder
            .enc_embedding
            .subject_embedding
            .subject_embedding
        )
        subject_id = embedding_layer.num_embeddings

    return torch.full(
        (batch_size,),
        subject_id,
        dtype=torch.long,
        device=device,
    )

def _compute_rsa_loss(eeg_features, img_features, eps=1e-8,):
    """
    Compute RSA loss between EEG-predicted features and image CLIP features.

    eeg_features: [B, 1024]
    img_features: [B, 1024]

    return:
        scalar RSA loss = 1 - Pearson correlation
    """

    batch_size = eeg_features.size(0)

    # Pearson相関を取るには複数の刺激間ペアが必要
    if batch_size < 3:
        return eeg_features.sum() * 0.0

    # ① 各特徴ベクトルをL2 normalize
    eeg_n = F.normalize(eeg_features, dim=-1)
    img_n = F.normalize(img_features, dim=-1)

    # ② cosine distance RDMを作る
    # shape: [B, B]
    eeg_rdm = 1.0 - eeg_n @ eeg_n.T
    img_rdm = 1.0 - img_n @ img_n.T

    # ③ RDMの上三角（対角成分を除く）のindexを取得
    tri = torch.triu_indices(
        eeg_rdm.shape[0],
        eeg_rdm.shape[0],
        offset=1,
        device=eeg_features.device,
    )

    # ④ B×BのRDMから刺激ペア部分だけ1次元に取り出す
    eeg_vec = eeg_rdm[tri[0], tri[1]]
    img_vec = img_rdm[tri[0], tri[1]]

    # ⑤ Pearson相関用に平均を引く
    eeg_centered = eeg_vec - eeg_vec.mean()
    img_centered = img_vec - img_vec.mean() 
    
    # ⑥ Pearson相関の分子
    numerator = torch.sum(
        eeg_centered * img_centered
    )

    # ⑦ Pearson相関の分母
    denominator = (
        torch.sqrt(torch.sum(eeg_centered ** 2) + eps)
        *
        torch.sqrt(torch.sum(img_centered ** 2) + eps)
    )

    # ⑧ RSA
    rsa = numerator / denominator

    # RSAを最大化したいので、lossとしては1-RSA
    rsa_loss = 1.0 - rsa

    return rsa_loss

def _compute_rdm_mse_loss(eeg_features, img_features):
    batch_size = eeg_features.size(0)

    if batch_size < 2:
        return eeg_features.sum() * 0.0

    eeg_n = F.normalize(eeg_features, dim=-1)
    img_n = F.normalize(img_features, dim=-1)

    eeg_rdm = 1.0 - eeg_n @ eeg_n.T
    img_rdm = 1.0 - img_n @ img_n.T

    tri = torch.triu_indices(
        batch_size,
        batch_size,
        offset=1,
        device=eeg_features.device,
    )

    eeg_dist = eeg_rdm[tri[0], tri[1]]
    img_dist = img_rdm[tri[0], tri[1]]

    return F.mse_loss(eeg_dist, img_dist)

def _compute_loss(eeg_features, img_features, logit_scale, loss_func,
                  loss_mode: str, alpha: float,
                  rsa_weight: float = 0.0,
                  rsa_loss_type: str = 'pearson',
                  return_components: bool = False,
                  text_features=None):
    """Return scalar loss for one batch."""
    if loss_mode == 'generation':
        mse = nn.functional.mse_loss(eeg_features, img_features)
        eeg_n = F.normalize(eeg_features, dim=-1)
        img_n = F.normalize(img_features, dim=-1)
        clip_loss = loss_func(eeg_n, img_n, logit_scale)

        mse_term = alpha * mse * 10
        clip_term = (1 - alpha) * clip_loss * 10

        rsa_loss = eeg_features.new_tensor(0.0)
        rsa_term = eeg_features.new_tensor(0.0)

        if rsa_weight > 0.0:
            if rsa_loss_type == 'pearson':
                rsa_loss = _compute_rsa_loss(eeg_features, img_features)

            elif rsa_loss_type == 'rdm_mse':
                rsa_loss = _compute_rdm_mse_loss(eeg_features, img_features)

            else:
                raise ValueError(
                f"Unknown rsa_loss_type: {rsa_loss_type}"
            )
            
            rsa_term = rsa_weight * rsa_loss

        total_loss = mse_term + clip_term + rsa_term

        if return_components:
            components = {
                "mse": mse.detach(),
                "clip": clip_loss.detach(),
                "rsa": rsa_loss.detach(),
                "mse_term": mse_term.detach(),
                "clip_term": clip_term.detach(),
                "rsa_term": rsa_term.detach(),
            }
            return total_loss, components

        return total_loss

    elif loss_mode == 'retrieval':
        eeg_n = F.normalize(eeg_features, dim=-1)
        img_n = F.normalize(img_features, dim=-1)
        img_loss = loss_func(eeg_n, img_n, logit_scale)
        if text_features is not None:
            txt_n = F.normalize(text_features, dim=-1)
            text_loss = loss_func(eeg_n, txt_n, logit_scale)
        else:
            text_loss = img_loss
        return alpha * img_loss + (1 - alpha) * text_loss

    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode!r}. "
                         "Choose 'generation' or 'retrieval'.")


def _logits_for_accuracy(eeg_features, img_pool, logit_scale, loss_mode: str):
    """Return similarity logits (B, n_pool) for top-1 accuracy tracking."""
    if loss_mode == 'retrieval':
        eeg_n   = F.normalize(eeg_features, dim=-1)
        img_n   = F.normalize(img_pool, dim=-1)
        return logit_scale * eeg_n @ img_n.T
    else:
        return logit_scale * eeg_features @ img_pool.T


# ── Data split ────────────────────────────────────────────────────────────────

def stratified_condition_split(n_classes: int = 1654,
                               conditions_per_class: int = 10,
                               trials_per_condition: int = 4,
                               val_ratio: float = 0.1,
                               seed: int = 42):
    """
    Stratified 9:1 training / validation split by *condition* within each class.

    Data layout per subject (66 160 total samples):
        class c: [cond0_trial0 cond0_trial1 cond0_trial2 cond0_trial3
                  cond1_trial0 ... cond9_trial3]
    Each class occupies ``conditions_per_class × trials_per_condition`` = 40 entries.

    Returns
    -------
    train_indices, val_indices  (sorted lists of integer sample indices)
    """
    rng = np.random.RandomState(seed)
    total_per_class = conditions_per_class * trials_per_condition

    n_val_conds = max(1, int(round(conditions_per_class * val_ratio)))

    train_indices, val_indices = [], []
    for cls in range(n_classes):
        base = cls * total_per_class
        conds = list(range(conditions_per_class))
        rng.shuffle(conds)
        val_conds = set(conds[:n_val_conds])

        for c in range(conditions_per_class):
            for t in range(trials_per_condition):
                idx = base + c * trials_per_condition + t
                (val_indices if c in val_conds else train_indices).append(idx)

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


# ── Training epoch ────────────────────────────────────────────────────────────

def train_encoder_epoch(sub, model, loader, optimizer, device,
                        img_features_all, *,
                        loss_mode: str,
                        alpha: float,
                        rsa_weight: float = 0.0,
                        rsa_loss_type: str = 'pearson',
                        use_subject_id: bool = True,
                        text_features_all=None):
    """
    Train the ATMS encoder for one epoch.

    Parameters
    ----------
    sub               : subject ID string, e.g. 'sub-08'
    model             : ATMS model (must have .logit_scale and .loss_func)
    loader            : DataLoader yielding (eeg, labels, text, text_feats, img, img_feats)
    optimizer         : torch optimizer
    device            : torch device
    img_features_all  : (N, D) full training image feature bank (for accuracy computation)
    loss_mode         : 'generation' or 'retrieval'
    alpha             : loss blending weight
    text_features_all : (N, D) full training text feature bank
                        (only used in 'retrieval' mode; ignored otherwise)

    Returns
    -------
    avg_loss : float
    accuracy : float  (top-1 k-way accuracy against full training feature bank)
    """
    model.train()
    img_pool = img_features_all.to(device).float()
    # In generation mode the pool is every-10th sample (one per class)
    # if loss_mode == 'generation':
    #     img_pool = img_pool[::10]

    # subject_id = _extract_subject_id(sub)
    total_loss = 0.0

    component_sums = None
    if loss_mode == 'generation':
        component_sums = {
            "mse": 0.0,
            "clip": 0.0,
            "rsa": 0.0,
            "mse_term": 0.0,
            "clip_term": 0.0,
            "rsa_term": 0.0,
        }
    correct = 0
    total = 0

    for batch_idx, (eeg_data, labels, text, text_feats, img, img_feats) in enumerate(loader):
        eeg_data  = eeg_data.to(device)
        img_feats = img_feats.to(device).float()
        labels    = labels.to(device)
        txt_feats = text_feats.to(device).float() if loss_mode == 'retrieval' else None

        optimizer.zero_grad()

        batch_size  = eeg_data.size(0)
        subject_ids = _make_subject_ids(
        sub=sub,
        model=model,
        batch_size=batch_size,
        device=device,
        use_subject_id=use_subject_id,
        )
        eeg_features = model(eeg_data, subject_ids).float()

        loss_output = _compute_loss(
            eeg_features, img_feats, model.logit_scale,
            model.loss_func, loss_mode, alpha,
            rsa_weight=rsa_weight,
            rsa_loss_type=rsa_loss_type,
            return_components=(loss_mode == 'generation'),
            text_features=txt_feats)

        if loss_mode == 'generation':
            loss, components = loss_output
        else:
            loss = loss_output
            components = None

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if components is not None and component_sums is not None:
            for name, value in components.items():
                component_sums[name] += value.item()

        # Top-1 accuracy against full feature bank
        with torch.no_grad():
            logits    = _logits_for_accuracy(eeg_features.detach(), img_pool,
                                              model.logit_scale, loss_mode)
            predicted = torch.argmax(logits, dim=1)
            total    += batch_size
            correct  += (predicted == labels).sum().item()

        del eeg_data, eeg_features, img_feats

    num_batches = batch_idx + 1

    avg_components = None
    if component_sums is not None:
        avg_components = {
            name: value / num_batches
            for name, value in component_sums.items()
        }

    return total_loss / num_batches, correct / total, avg_components


# ── Validation / evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_encoder(sub, model, loader, device, img_features_all, *,
                     k: int = 200,
                     loss_mode: str,
                     alpha: float,
                     rsa_weight: float = 0.0,
                     rsa_loss_type: str = "pearson",
                     use_subject_id: bool = True,
                     text_features_all=None):
    """
    Evaluate the ATMS encoder on a validation or test split.

    For each sample, ``k``-way retrieval accuracy is computed by randomly
    sampling ``k-1`` distractors from the feature bank.

    Returns
    -------
    avg_loss : float
    k_way_accuracy : float
    """
    model.eval()
    img_pool = img_features_all.to(device).float()
    all_classes = set(range(img_pool.size(0)))

    # subject_id = _extract_subject_id(sub)
    total_loss = 0.0

    component_sums = None
    if loss_mode == 'generation':
        component_sums = {
            "mse": 0.0,
            "clip": 0.0,
            "rsa": 0.0,
            "mse_term": 0.0,
            "clip_term": 0.0,
            "rsa_term": 0.0,
        }

    correct = 0
    total = 0

    all_eeg_features = []
    all_img_features = []

    for batch_idx, (eeg_data, labels, text, text_feats, img, img_feats) in enumerate(loader):
        eeg_data  = eeg_data.to(device)
        img_feats = img_feats.to(device).float()
        labels    = labels.to(device)
        txt_feats = text_feats.to(device).float() if loss_mode == 'retrieval' else None

        batch_size  = eeg_data.size(0)
        subject_ids = _make_subject_ids(
        sub=sub,
        model=model,
        batch_size=batch_size,
        device=device,
        use_subject_id=use_subject_id,
        )
        eeg_features = model(eeg_data, subject_ids).float()

        if loss_mode == 'generation':
            all_eeg_features.append(eeg_features.detach().cpu())
            all_img_features.append(img_feats.detach().cpu())

        loss_output = _compute_loss(
            eeg_features, img_feats, model.logit_scale,
            model.loss_func, loss_mode, alpha,
            rsa_weight=rsa_weight,
            rsa_loss_type=rsa_loss_type,
            return_components=(loss_mode == 'generation'),
            text_features=txt_feats)

        if loss_mode == 'generation':
            loss, components = loss_output
        else:
            loss = loss_output
            components = None

        total_loss += loss.item()

        if components is not None and component_sums is not None:
            for name, value in components.items():
                component_sums[name] += value.item()

        # k-way retrieval
        for i, label in enumerate(labels):
            possible  = list(all_classes - {label.item()})
            selected  = random.sample(possible, min(k - 1, len(possible))) + [label.item()]
            sel_feats = img_pool[selected]
            logits    = _logits_for_accuracy(
                eeg_features[i].unsqueeze(0), sel_feats,
                model.logit_scale, loss_mode).squeeze(0)
            pred = selected[torch.argmax(logits).item()]
            if pred == label.item():
                correct += 1
            total += 1

        del eeg_data, eeg_features, img_feats

    num_batches = batch_idx + 1

    avg_components = None
    if component_sums is not None:
        avg_components = {
            name: value / num_batches
            for name, value in component_sums.items()
        }

    val_rsa = None

    if loss_mode == 'generation' and all_eeg_features:
        eeg_features_full = torch.cat(all_eeg_features, dim=0)
        img_features_full = torch.cat(all_img_features, dim=0)

        rsa_loss_full = _compute_rsa_loss(
            eeg_features_full,
            img_features_full,
        )
        val_rsa = (1.0 - rsa_loss_full).item()
    
    return (
        total_loss / num_batches,
        correct / total,
        avg_components,
        val_rsa,
    )
