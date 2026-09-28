"""Dataset loading and splitting for train.py/evaluate.py.

Two split modes:

  - "random": reproduces notebook/KSL.ipynb Cell 6 exactly -- a single
    stratified-by-class 80/20 train/test split (seed 42), then a further
    stratified 90/10 train/val split of the 80%. No signer grouping,
    because the legacy `.pkl` this mode is meant to run against
    (data/KSL77_joint_stream_47pt.pkl) never recorded signer identity in
    the first place (see AUDIT.md §4).

  - "signer_kfold": GroupKFold(n_splits=5) grouped by signer_id, so each of
    the 20 KSL-77 signers' videos land in exactly one test fold. Requires a
    manifest with a signer_id column (i.e. data produced by
    extract_landmarks.py, not the legacy pkl). A validation subset of
    signers is carved out of each fold's training signers via
    GroupShuffleSplit, so validation never touches that fold's held-out
    test signers either.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Optional

import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset

import transforms


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_legacy_pkl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads data/KSL77_joint_stream_47pt.pkl -> (features (N,3,32,47), labels (N,))."""
    with open(path, "rb") as f:
        features, labels = pickle.load(f)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_manifest_dataset(
    features_path: str, mask_path: str, manifest_path: str, min_hand_frac: float = 0.0
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Loads output of extract_landmarks.py -> (features, mask, manifest_df),
    with drop_undetected_samples() applied so a corrupted/undecodable video
    (e.g. "moov atom not found") never silently becomes a real training or
    test example. See drop_undetected_samples for exactly what's dropped.
    """
    features = np.load(features_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.uint8)
    manifest = pd.read_csv(manifest_path, dtype={"signer_id": str})
    assert len(manifest) == len(features) == len(mask), (
        f"length mismatch: manifest={len(manifest)} features={len(features)} mask={len(mask)}"
    )
    return drop_undetected_samples(features, mask, manifest, min_hand_frac=min_hand_frac)


def drop_undetected_samples(
    features: np.ndarray,
    mask: np.ndarray,
    manifest: pd.DataFrame,
    min_hand_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Drops rows extract_landmarks.py effectively couldn't extract anything
    from -- most importantly a corrupted/truncated video (e.g. cv2/ffmpeg's
    "moov atom not found") that decodes zero frames and would otherwise be
    trained on and tested as a legitimate, all-zero-landmark example.

    Filters `features`, `mask`, and `manifest` together, in lockstep, so row
    order/alignment is preserved, and prints exactly which video_path(s)
    were dropped and why.

    A row is dropped if any of:
      - n_sampled_ok == 0        (nothing decoded at all -- corrupt/unreadable file)
      - frac_frames_with_pose == 0   (MediaPipe never found a subject in any sampled frame)
      - frac_frames_with_any_hand < min_hand_frac   (opt-in via --min-hand-frac; default 0.0 = off)
    """
    n = len(manifest)
    if n == 0:
        return features, mask, manifest

    if "n_sampled_ok" in manifest.columns:
        zero_decoded = (manifest["n_sampled_ok"] == 0).to_numpy()
    else:
        print(
            "dataset.py WARNING: manifest.csv has no 'n_sampled_ok' column "
            "(from an extract_landmarks.py run before this check existed) -- "
            "can't detect fully-undecodable videos this way. Re-run "
            "extract_landmarks.py against the same --out to regenerate "
            "manifest.csv from the existing cache/ (no MediaPipe re-run "
            "needed) and pick up this column."
        )
        zero_decoded = np.zeros(n, dtype=bool)

    zero_pose = (manifest["frac_frames_with_pose"] == 0).to_numpy()

    if min_hand_frac > 0.0 and "frac_frames_with_any_hand" in manifest.columns:
        below_hand_frac = manifest["frac_frames_with_any_hand"].to_numpy() < min_hand_frac
    else:
        below_hand_frac = np.zeros(n, dtype=bool)

    drop = zero_decoded | zero_pose | below_hand_frac
    if drop.any():
        print(f"dataset.py: dropping {int(drop.sum())} of {n} sample(s) before splitting:")
        for i in np.where(drop)[0]:
            reasons = []
            if zero_decoded[i]:
                reasons.append("n_sampled_ok == 0 (undecodable video)")
            if zero_pose[i]:
                reasons.append("frac_frames_with_pose == 0")
            if below_hand_frac[i]:
                frac = manifest["frac_frames_with_any_hand"].iloc[i]
                reasons.append(
                    f"frac_frames_with_any_hand={frac:.2f} < --min-hand-frac={min_hand_frac}"
                )
            print(f"  - {manifest['video_path'].iloc[i]}: {'; '.join(reasons)}")

    keep = ~drop
    return features[keep], mask[keep], manifest.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Label re-indexing (dense 0..C-1 <-> original class ids), shared by both split modes
# ---------------------------------------------------------------------------


def build_label_maps(labels: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    unique_labels = sorted(int(l) for l in np.unique(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    reverse_label_map = {new: old for old, new in label_map.items()}
    return label_map, reverse_label_map


# ---------------------------------------------------------------------------
# Split result containers
# ---------------------------------------------------------------------------


@dataclass
class Split:
    """One train/val/test split, in dense re-indexed label space."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    label_map: dict[int, int]
    reverse_label_map: dict[int, int]
    # signer_kfold only; empty for "random"
    fold: Optional[int] = None
    train_signers: set = field(default_factory=set)
    val_signers: set = field(default_factory=set)
    test_signers: set = field(default_factory=set)


def _dense(labels: np.ndarray, label_map: dict[int, int]) -> np.ndarray:
    return np.array([label_map[int(l)] for l in labels], dtype=np.int64)


# ---------------------------------------------------------------------------
# "random" split (reproduces the notebook exactly)
# ---------------------------------------------------------------------------


def random_split(labels: np.ndarray, seed: int = 42) -> Split:
    n = len(labels)
    all_idx = np.arange(n)
    label_map, reverse_label_map = build_label_maps(labels)

    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, stratify=labels, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, stratify=labels[train_idx], random_state=seed
    )
    return Split(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        label_map=label_map,
        reverse_label_map=reverse_label_map,
    )


# ---------------------------------------------------------------------------
# "signer_kfold" split
# ---------------------------------------------------------------------------


def signer_kfold_splits(
    labels: np.ndarray,
    signer_ids: np.ndarray,
    n_splits: int = 5,
    val_frac_of_train_signers: float = 0.15,
    seed: int = 42,
) -> list[Split]:
    n = len(labels)
    all_idx = np.arange(n)
    label_map, reverse_label_map = build_label_maps(labels)
    signer_ids = np.asarray(signer_ids)

    gkf = GroupKFold(n_splits=n_splits)
    folds: list[Split] = []
    for fold, (train_val_idx, test_idx) in enumerate(
        gkf.split(all_idx, labels, groups=signer_ids)
    ):
        test_signers = set(signer_ids[test_idx])

        # Carve a validation set of *signers* out of this fold's training
        # signers (never touching test_signers).
        train_val_signer_ids = signer_ids[train_val_idx]
        gss = GroupShuffleSplit(
            n_splits=1, test_size=val_frac_of_train_signers, random_state=seed + fold
        )
        train_pos, val_pos = next(
            gss.split(train_val_idx, groups=train_val_signer_ids)
        )
        train_idx = train_val_idx[train_pos]
        val_idx = train_val_idx[val_pos]

        train_signers = set(signer_ids[train_idx])
        val_signers = set(signer_ids[val_idx])

        assert train_signers.isdisjoint(test_signers)
        assert val_signers.isdisjoint(test_signers)
        assert val_signers.isdisjoint(train_signers)

        folds.append(
            Split(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                label_map=label_map,
                reverse_label_map=reverse_label_map,
                fold=fold,
                train_signers=train_signers,
                val_signers=val_signers,
                test_signers=test_signers,
            )
        )

    check_signer_partition(folds, signer_ids)
    return folds


def check_signer_partition(folds: list[Split], signer_ids: np.ndarray) -> None:
    """Every signer appears in exactly one fold's test set, and within each
    fold, train/val/test signer sets never overlap.

    Raises AssertionError on any violation. Called both when folds are built
    here and again, independently, by evaluate.py before it reports numbers.
    """
    all_signers = set(np.asarray(signer_ids).tolist())
    test_signer_sets = [f.test_signers for f in folds]

    for f in folds:
        assert f.train_signers.isdisjoint(f.test_signers), (
            f"fold {f.fold}: train/test signer overlap: "
            f"{f.train_signers & f.test_signers}"
        )
        assert f.val_signers.isdisjoint(f.test_signers), (
            f"fold {f.fold}: val/test signer overlap: {f.val_signers & f.test_signers}"
        )
        assert f.val_signers.isdisjoint(f.train_signers), (
            f"fold {f.fold}: val/train signer overlap: {f.val_signers & f.train_signers}"
        )

    seen = set()
    for s in test_signer_sets:
        overlap = seen & s
        assert not overlap, f"signer(s) {overlap} appear in more than one test fold"
        seen |= s

    missing = all_signers - seen
    assert not missing, f"signer(s) {missing} never appear in any test fold"


# ---------------------------------------------------------------------------
# torch Dataset
# ---------------------------------------------------------------------------


class PoseDataset(Dataset):
    """Same augmentations as notebook/KSL.ipynb's PoseDataset (Cell 6), plus
    an optional --mirror-aug (see transforms.mirror_clip).

    `mask` (N, T, J), if given, is the real per-joint detection mask from
    extract_landmarks.py, used so mirroring reflects/swaps only actually-
    detected joints correctly. If `mask` is None (e.g. the legacy .pkl,
    which has no mask at all), a joint is treated as "detected" in a frame
    iff its (x, y, z) isn't exactly (0, 0, 0) -- the same silent-zero-fill
    convention that data was written with, so this is the best available
    stand-in, not a new assumption.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray | None = None,
        augment: bool = False,
        mirror_aug: bool = False,
        mirror_p: float = 0.5,
    ):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.mask = mask
        self.augment = augment
        self.mirror_aug = mirror_aug
        self.mirror_p = mirror_p

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.mirror_aug and random.random() < self.mirror_p:
            x = self._mirror(x, idx)
        if self.augment:
            x = self.apply_augmentations(x)
        return x, y

    def _mirror(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        coords = transforms.chw_to_tjc(x.numpy())
        if self.mask is not None:
            joint_mask = self.mask[idx].astype(bool)
        else:
            joint_mask = transforms.infer_mask_from_coords(coords)
        mirrored_coords, _ = transforms.mirror_clip(coords, joint_mask)
        return torch.as_tensor(transforms.tjc_to_chw(mirrored_coords), dtype=torch.float32)

    def apply_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x += torch.randn_like(x) * 0.01
        if random.random() < 0.3:
            x *= 1.0 + (random.random() - 0.5) * 0.1
        if random.random() < 0.3:
            x = torch.roll(x, shifts=random.randint(-2, 2), dims=1)
        if random.random() < 0.3:
            x[0] = -x[0]
        return x


def build_datasets(
    features: np.ndarray,
    labels: np.ndarray,
    split: Split,
    mask: np.ndarray | None = None,
    augment_train: bool = True,
    mirror_aug: bool = False,
    mirror_p: float = 0.5,
) -> tuple[PoseDataset, PoseDataset, PoseDataset]:
    y_dense = _dense(labels, split.label_map)
    train_mask = mask[split.train_idx] if mask is not None else None
    train_ds = PoseDataset(
        features[split.train_idx],
        y_dense[split.train_idx],
        mask=train_mask,
        augment=augment_train,
        mirror_aug=mirror_aug,
        mirror_p=mirror_p,
    )
    val_ds = PoseDataset(features[split.val_idx], y_dense[split.val_idx], augment=False)
    test_ds = PoseDataset(features[split.test_idx], y_dense[split.test_idx], augment=False)
    return train_ds, val_ds, test_ds
