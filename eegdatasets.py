"""
Unified THINGS-EEG2 dataset loader — shared by Generation and Retrieval.

Feature modes (``feature_type`` argument)
------------------------------------------
  'ViT-H-14' / 'clip'
      Load (or lazily compute) OpenCLIP ViT-H-14 text + image features.
      Features are cached to ``features_dir`` when computed.
      Override the cache file with ``features_path``.

  'UniLIP-2B' / 'UniLIP-3B'
      Load pre-computed UniLIP vision features.  CLS token (index 0) is
      extracted automatically.  Default path searched first; override with
      ``features_path``.

  'InternVL3-2B'
      Load pre-computed InternVL3 features (key: 'cls_features').

  'EVA01_CLIP_g_14_plus'
      Load EVA-CLIP embeddings (key: 'embeddings').

  'vae_latent'
      Load pre-computed SDXL-VAE image latents (4 × 64 × 64).
      Expects ``<latent_dir>/train_image_latent_512.pt`` / ``test_…``.

Raw (unnormalized) features are always stored.
**L2 normalization must be applied in the training/eval loop** before
computing contrastive loss — not inside this dataset.

Shortcut: pass ``preloaded_features={'text_features': …, 'img_features': …}``
to skip all feature loading and use the supplied tensors directly.  This is
the preferred pattern for Retrieval scripts that load features once outside
the per-subject loop.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn import functional as F
from PIL import Image

# ── Shared feature cache directory ────────────────────────────────────────────
# Both Generation and Retrieval scripts read/write CLIP features here so that
# each model only needs to be computed once, regardless of which subdirectory
# the training script is launched from.
# Features are stored **raw / unnormalized**; callers apply L2 normalisation
# inside the loss computation only — never before persisting to disk.
_DEFAULT_FEATURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'features')

# ── Default feature-file paths for non-CLIP encoders ─────────────────────────
_FEATURE_DEFAULTS = {
    'UniLIP-2B': {
        True:  '/vePFS-0x0d/home/atm-2/features/UniLIP-2B_vision_features_train.pt',
        False: '/vePFS-0x0d/home/atm-2/features/UniLIP-2B_vision_features_test.pt',
        'img_key':   'features',
        'img_slice': lambda x: x[:, 0, :],   # CLS token
    },
    'UniLIP-3B': {
        True:  '/vePFS-0x0d/home/atm-2/features/UniLIP-3B_vision_features_train.pt',
        False: '/vePFS-0x0d/home/atm-2/features/UniLIP-3B_vision_features_test.pt',
        'img_key':   'features',
        'img_slice': lambda x: x[:, 0, :],
    },
    'InternVL3-2B': {
        True:  '/vePFS-0x0d/home/atm-2/features/InternVL3-2B_vision_features_train.pt',
        False: '/vePFS-0x0d/home/atm-2/features/InternVL3-2B_vision_features_test.pt',
        'img_key':   'cls_features',
        'img_slice': None,
    },
    'EVA01_CLIP_g_14_plus': {
        True:  '/vePFS-0x0d/home/atm-2/features/evaclip/training_images_eva_clip_embeddings.pt',
        False: '/vePFS-0x0d/home/atm-2/features/evaclip/test_images_eva_clip_embeddings.pt',
        'img_key':   'embeddings',
        'img_slice': None,
    },
}

# ── Lazy OpenCLIP loader ──────────────────────────────────────────────────────
_CLIP_MODEL_TYPE = 'ViT-H-14'
_clip_state: dict = {}


def _ensure_clip_loaded():
    if 'model' in _clip_state:
        return
    import clip
    import open_clip
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cache_dir = os.environ.get('OPEN_CLIP_CACHE_DIR', None)
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        _CLIP_MODEL_TYPE,
        pretrained="laion2b_s32b_b79k",
        precision="fp32",
        device=device,
        cache_dir=cache_dir,
    )

    model.eval()

    _clip_state["model"] = model
    _clip_state["preprocess"] = preprocess_val
    _clip_state['device'] = device
    _clip_state['clip'] = clip
    print(f"OpenCLIP {_CLIP_MODEL_TYPE} loaded on {device}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    """
    THINGS-EEG2 EEG dataset.

    Args:
        data_path:          Root of preprocessed EEG per-subject folders.
        img_dir_training:   Directory with training class sub-folders.
        img_dir_test:       Directory with test class sub-folders.
        feature_type:       One of 'ViT-H-14'/'clip', 'UniLIP-2B', 'UniLIP-3B',
                            'InternVL3-2B', 'EVA01_CLIP_g_14_plus', 'vae_latent'.
        features_dir:       Directory to look for / cache ViT-H-14 .pt files.
        features_path:      Explicit path to a pre-computed feature .pt file
                            (overrides defaults for all non-CLIP feature types).
        latent_dir:         Directory with train/test_image_latent_512.pt
                            (used only when feature_type='vae_latent').
        preloaded_features: Dict ``{'text_features': Tensor, 'img_features': Tensor}``
                            bypasses all feature loading — use this in Retrieval
                            scripts to avoid re-loading features per subject.
        exclude_subject:    Subject to exclude from training split.
        subjects:           Subject IDs to include (default: all found in data_path).
        train:              True = training split, False = test split.
        time_window:        [start, end] seconds for EEG time-axis crop.
        classes:            Optional list of class indices to select.
        pictures:           Optional list of picture indices (paired with classes).
    """

    def __init__(self, data_path,
                 img_dir_training=None,
                 img_dir_test=None,
                 feature_type='ViT-H-14',
                 features_dir=None,
                 features_path=None,
                 latent_dir='.',
                 preloaded_features=None,
                 exclude_subject=None,
                 subjects=None,
                 train=True,
                 time_window=None,
                 classes=None,
                 pictures=None,
                 avg_trials=False):

        if time_window is None:
            time_window = [0, 1.0]

        self.data_path = data_path
        self.img_dir_training = img_dir_training
        self.img_dir_test = img_dir_test
        self.feature_type = feature_type
        # Use the shared cross-script cache directory when not overridden
        self.features_dir = features_dir if features_dir else _DEFAULT_FEATURES_DIR
        self.features_path = features_path
        self.latent_dir = latent_dir
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject
        self.avg_trials = avg_trials and train

        assert any(s in self.subject_list for s in self.subjects), \
            f"None of {self.subjects} found in {data_path}"

        self.data, self.labels, self.text, self.img = self._load_eeg_and_images()
        self.data = self._extract_time_window(self.data, time_window)
        
        self.visual_projection = None
        
        # ── Feature loading ──────────────────────────────────────────────
        if preloaded_features is not None:
            # Fastest path: caller already loaded features once outside loop
            self.text_features = preloaded_features.get('text_features')
            self.img_features = preloaded_features['img_features']
        elif self.classes is not None or self.pictures is not None:
            # Subset mode: compute features on the fly
            _ensure_clip_loaded()
            self.text_features = self._encode_text(self.text)
            self.img_features = self._encode_images(self.img)
        else:
            self._load_features()

    # ── Feature loading ───────────────────────────────────────────────────────

    def _load_features(self):
        ft = self.feature_type

        if ft in ('ViT-H-14', 'clip'):
            self._load_clip_features()

        elif ft == 'vae_latent':
            fname = ('train_image_latent_512.pt' if self.train
                     else 'test_image_latent_512.pt')
            candidates = [
                os.path.join(self.latent_dir, fname),
                self.features_path,
                fname,
            ]
            load_from = next((p for p in candidates
                              if p and os.path.exists(p)), None)
            if load_from is None:
                raise FileNotFoundError(
                    f"VAE latent file not found (looked in {self.latent_dir}). "
                    "Pre-compute with extract_vae_latents.py first.")
            saved = torch.load(load_from, weights_only=False)
            self.text_features = None
            self.img_features = saved['image_latent']
            print(f"Loaded VAE latents from {load_from}  shape={self.img_features.shape}")

        elif ft in _FEATURE_DEFAULTS:
            meta = _FEATURE_DEFAULTS[ft]
            default_path = meta[self.train]
            load_from = (self.features_path
                         if self.features_path and os.path.exists(self.features_path)
                         else default_path)
            if not os.path.exists(load_from):
                raise FileNotFoundError(
                    f"{ft} feature file not found: {load_from}")
            saved = torch.load(load_from, weights_only=False)
            key = meta['img_key']
            img_feats = saved.get(key)
            if img_feats is None:
                raise KeyError(f"Key '{key}' not found in {load_from}")
            if meta['img_slice'] is not None:
                img_feats = meta['img_slice'](img_feats)
            self.img_features = img_feats
            self.text_features = saved.get('text_features', img_feats.clone())
            print(f"Loaded {ft} features from {load_from}  "
                  f"img shape={self.img_features.shape}")

        else:
            raise ValueError(
                f"Unknown feature_type: {ft!r}.  "
                f"Choose from: clip/ViT-H-14, vae_latent, "
                + ', '.join(_FEATURE_DEFAULTS))

    def _load_clip_features(self):
        fname = (
            f"{_CLIP_MODEL_TYPE}_preprojection_features_train.pt"
            if self.train
            else f"{_CLIP_MODEL_TYPE}_preprojection_features_test.pt"
        )

        candidates = [
            self.features_path,
            os.path.join(self.features_dir, fname),
            fname,
        ]

        load_from = next(
            (p for p in candidates if p and os.path.exists(p)),
            None,
        )

        if load_from is not None:
            print(f"Loading pre-extracted CLIP features from: {load_from}")

            saved = torch.load(
                load_from,
                map_location="cpu",
                weights_only=False,
            )

            self.text_features = saved["text_features"]
            self.img_features = saved["img_features"]
            self.visual_projection = saved.get("visual_projection")

            # 古いキャッシュにprojectionがない場合だけ追加
            if self.visual_projection is None:
                _ensure_clip_loaded()

                self.visual_projection = (
                    _clip_state["model"]
                    .visual.proj
                    .detach()
                    .float()
                    .cpu()
                )

                saved["visual_projection"] = self.visual_projection
                torch.save(saved, load_from)

        else:
            print("CLIP preprojection featuresを抽出します")

            _ensure_clip_loaded()

            self.text_features = self._encode_text(self.text)
            self.img_features = self._encode_images(self.img)

            self.visual_projection = (
                _clip_state["model"]
                .visual.proj
                .detach()
                .float()
                .cpu()
            )

            cache = os.path.join(self.features_dir, fname)
            os.makedirs(self.features_dir, exist_ok=True)

            torch.save(
                {
                    "text_features": self.text_features.cpu(),
                    "img_features": self.img_features.cpu(),
                    "visual_projection": self.visual_projection,
                },
                cache,
            )

            print(f"保存しました: {cache}")

        # CLIP本体は以降不要
        _clip_state.clear()

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Image features:", self.img_features.shape)
        print("Visual projection:", self.visual_projection.shape)
    
    def _load_eeg_and_images(self):
        data_list = []
        label_list = []
        texts = []
        images = []

        img_directory = (
            self.img_dir_training
            if self.train
            else self.img_dir_test
        )

        if img_directory is None:
            raise ValueError(
                f"{'img_dir_training' if self.train else 'img_dir_test'} "
                "must be provided."
            )

        all_folders = sorted(
            folder
            for folder in os.listdir(img_directory)
            if os.path.isdir(
                os.path.join(img_directory, folder)
            )
        )

        # テキストラベルを作成
        selected_folders = (
            [all_folders[i] for i in self.classes]
            if self.classes is not None
            else all_folders
        )

        for folder in selected_folders:
            try:
                class_name = folder[folder.index("_") + 1:]
                texts.append(f"This picture is {class_name}")
            except ValueError:
                pass

        # 画像ファイル一覧を作成
        def _list_images(folder_path):
            return sorted(
                filename
                for filename in os.listdir(folder_path)
                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
            )

        if (
            self.classes is not None
            and self.pictures is not None
        ):
            for class_index, picture_index in zip(
                self.classes,
                self.pictures,
            ):
                folder_path = os.path.join(
                    img_directory,
                    all_folders[class_index],
                )

                image_names = _list_images(folder_path)

                if picture_index < len(image_names):
                    images.append(
                        os.path.join(
                            folder_path,
                            image_names[picture_index],
                        )
                    )

        elif self.classes is not None:
            for class_index in self.classes:
                folder_path = os.path.join(
                    img_directory,
                    all_folders[class_index],
                )

                images.extend(
                    os.path.join(folder_path, filename)
                    for filename in _list_images(folder_path)
                )

        else:
            for folder in all_folders:
                folder_path = os.path.join(
                    img_directory,
                    folder,
                )

                images.extend(
                    os.path.join(folder_path, filename)
                    for filename in _list_images(folder_path)
                )

        print(
            f"Subjects: {self.subjects} "
            f"exclude: {self.exclude_subject}"
        )

        # EEGを読み込む
        for subject in self.subjects:
            if self.train:
                if subject == self.exclude_subject:
                    continue

                file_path = os.path.join(
                    self.data_path,
                    subject,
                    "preprocessed_eeg_training.npy",
                )

                data = np.load(
                    file_path,
                    allow_pickle=True,
                )

                eeg = torch.from_numpy(
                    data["preprocessed_eeg_data"]
                ).float().detach()

                print(eeg.shape)

                times = torch.from_numpy(
                    data["times"]
                ).detach()[50:]

                ch_names = data["ch_names"]

                n_classes = 1654
                conditions_per_class = 10

                if (
                    self.classes is not None
                    and self.pictures is not None
                ):
                    for class_index, picture_index in zip(
                        self.classes,
                        self.pictures,
                    ):
                        sample_index = (
                            class_index * 1
                            + picture_index
                        )

                        if sample_index < len(eeg):
                            data_list.append(
                                eeg[
                                    sample_index:
                                    sample_index + 1
                                ]
                            )

                            label_list.append(
                                torch.full(
                                    (1,),
                                    class_index,
                                    dtype=torch.long,
                                )
                            )

                elif self.classes is not None:
                    for class_index in self.classes:
                        start_index = (
                            class_index
                            * conditions_per_class
                        )

                        chunk = eeg[
                            start_index:
                            start_index
                            + conditions_per_class
                        ]

                        if self.avg_trials:
                            chunk = chunk.mean(dim=1)

                        data_list.append(chunk)

                        label_list.append(
                            torch.full(
                                (conditions_per_class,),
                                class_index,
                                dtype=torch.long,
                            )
                        )

                else:
                    for class_index in range(n_classes):
                        start_index = (
                            class_index
                            * conditions_per_class
                        )

                        chunk = eeg[
                            start_index:
                            start_index
                            + conditions_per_class
                        ]

                        if self.avg_trials:
                            chunk = chunk.mean(dim=1)

                        data_list.append(chunk)

                        label_list.append(
                            torch.full(
                                (conditions_per_class,),
                                class_index,
                                dtype=torch.long,
                            )
                        )

            else:
                if (
                    self.exclude_subject is not None
                    and subject != self.exclude_subject
                ):
                    continue

                file_path = os.path.join(
                    self.data_path,
                    subject,
                    "preprocessed_eeg_test.npy",
                )

                data = np.load(
                    file_path,
                    allow_pickle=True,
                )

                eeg = torch.from_numpy(
                    data["preprocessed_eeg_data"]
                ).float().detach()

                times = torch.from_numpy(
                    data["times"]
                ).detach()[50:]

                ch_names = data["ch_names"]

                for class_index in range(200):
                    if (
                        self.classes is not None
                        and class_index not in self.classes
                    ):
                        continue

                    chunk = eeg[
                        class_index:
                        class_index + 1
                    ]

                    chunk = torch.mean(
                        chunk.squeeze(0),
                        dim=0,
                    )

                    data_list.append(chunk)

                    label_list.append(
                        torch.full(
                            (1,),
                            class_index,
                            dtype=torch.long,
                        )
                    )

        if self.train:
            data_tensor = torch.cat(
                data_list,
                dim=0,
            )

            if not self.avg_trials:
                # (条件, 4 trials, C, T)
                # → (サンプル, C, T)
                data_tensor = data_tensor.view(
                    -1,
                    *data_tensor.shape[2:],
                )

        else:
            data_tensor = torch.cat(
                data_list,
                dim=0,
            ).view(
                -1,
                *data_list[0].shape,
            )

        label_tensor = torch.cat(
            label_list,
            dim=0,
        )

        if self.train:
            if not self.avg_trials:
                label_tensor = (
                    label_tensor.repeat_interleave(4)
                )

            if self.classes is not None:
                unique_labels = []

                for value in label_tensor.tolist():
                    if value not in unique_labels:
                        unique_labels.append(value)

                mapping = {
                    value: index
                    for index, value
                    in enumerate(unique_labels)
                }

                label_tensor = torch.tensor(
                    [
                        mapping[
                            value.item()
                            if hasattr(value, "item")
                            else value
                        ]
                        for value in label_tensor
                    ],
                    dtype=torch.long,
                )

        self.times = times
        self.ch_names = ch_names

        print(
            f"EEG: {data_tensor.shape} "
            f"labels: {label_tensor.shape} "
            f"texts: {len(texts)} "
            f"images: {len(images)}"
        )

        return (
            data_tensor,
            label_tensor,
            texts,
            images,
        )



    # ── Time-window extraction ────────────────────────────────────────────────

    def _extract_time_window(self, eeg_data, time_window):
        start, end = time_window
        idx = (self.times >= start) & (self.times <= end)
        return eeg_data[..., idx]

    # ── CLIP encoders (on-the-fly, only for ViT-H-14 mode) ───────────────────

    def _encode_text(self, texts):
        clip = _clip_state['clip']
        model = _clip_state['model']
        dev = _clip_state['device']
        inputs = torch.cat([clip.tokenize(t) for t in texts]).to(dev)
        with torch.no_grad():
            feats = model.encode_text(inputs)
        # Return raw (unnormalized) features; caller normalizes before loss
        return feats.detach()

    def _encode_images(self, image_paths):
        model = _clip_state["model"]
        preprocess = _clip_state["preprocess"]
        dev = _clip_state["device"]

        feats_list = []

        def hook_fn(module, inputs, output):
            features = output.detach()

            if features.ndim == 3:
                features = features[:, 0]

            feats_list.append(features.float().cpu())

        handle = model.visual.ln_post.register_forward_hook(hook_fn)

        try:
            batch_size = 1

            for i in tqdm(
                range(0, len(image_paths), batch_size),
                desc="CLIP preprojection",
            ):
                batch = image_paths[i:i + batch_size]

                imgs = torch.stack([
                    preprocess(Image.open(p).convert("RGB"))
                    for p in batch
                ]).to(dev)

                with torch.no_grad():
                    model.encode_image(imgs)

        finally:
            handle.remove()

        features = torch.cat(feats_list, dim=0)
        print("Preprojection features:", features.shape)

        return features

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        label = self.labels[index]

        tpc = 1 if self.avg_trials else 4
        if self.pictures is None:
            n_cls = len(self.classes) if self.classes else self.n_cls
            n_train = n_cls * 10 * tpc
            n_test = n_cls * 1 * 80
        else:
            n_cls = len(self.classes) if self.classes else self.n_cls
            n_train = n_cls * 1 * tpc
            n_test = n_cls * 1 * 80

        if self.train:
            per_class = 10 * tpc if self.pictures is None else 1 * tpc
            text_idx = (index % n_train) // per_class
            img_idx = (index % n_train) // tpc
        else:
            text_idx = index % n_test
            img_idx = index % n_test

        text = self.text[text_idx]
        img = self.img[img_idx]
        text_feats = (self.text_features[text_idx]
                      if self.text_features is not None else -1)
        img_feats = self.img_features[img_idx]

        return x, label, text, text_feats, img, img_feats
