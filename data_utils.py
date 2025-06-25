from __future__ import annotations
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import joblib
import seaborn as sns 
import umap

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix,roc_auc_score
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit
from skimage import exposure
from skimage.filters import gaussian


BATCH_SIZE = 64
IMG_SIZE = (69, 69)
TARGET_SIZE = (224, 224)


# description in the docstring, used once at the top of the notebook
def inspect_and_plot_h5(h5_path, img_size=(69, 69)):
    """
    Open the HDF5 file, print dataset info and class counts,
    then plot one random image per class.
    """
    with h5py.File(h5_path, 'r') as f:
        # Assume datasets named 'images' and 'labels'
        images = f['images'][:]  # shape: (N, H, W, C)
        labels = f['ans'][:]  # shape: (N,)
    
    classes, counts = np.unique(labels, return_counts=True)
    print(f"Dataset contains {images.shape[0]} images of size {images.shape[1:]} and {len(classes)} classes")
    for cls, cnt in zip(classes, counts):
        print(f"  Class {int(cls)}: {cnt} images")
    
    # Plot one random image per class
    n = len(classes)
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols*3, rows*3))
    for i, cls in enumerate(classes):
        idx = np.random.choice(np.where(labels == cls)[0])
        img = images[idx]
        # Resize for display clarity
        img_resized = tf.image.resize(img, img_size).numpy().astype('uint8')
        
        plt.subplot(rows, cols, i+1)
        plt.imshow(img_resized)
        plt.title(f"Class {int(cls)}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    


# dataset reader for the classification task
def get_datasets(h5_path,
                 val_split: float = 0.2,
                 test_split: float = 0.1,
                 seed: int = 123):
    """
    Stratified split **train / val / test** and return **two** histograms:

    Returns
    -------
    train_ds_raw, val_ds_raw, test_ds_raw,
    class_weights,      # dict   - weights for TRAIN set
    class_counts_train, # dict   - counts *inside* TRAIN set (for oversampling)
    class_counts_full   # dict   - counts in the FULL dataset (for plots)
    """
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    # ---------- load whole file ----------
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]   # (N,H,W,C)
        labels = f["ans"][:]

    # -- full-dataset histogram (for bar-plot) -----------------
    classes_all, counts_all = np.unique(labels, return_counts=True)
    class_counts_full = dict(zip(classes_all.astype(int), counts_all))

    # ---------- stratified global shuffle ----------
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(labels))
    images, labels = images[idx], labels[idx]

    n_total = len(labels)
    n_test  = int(n_total * test_split)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    train_imgs, train_lbls = images[:n_train], labels[:n_train]
    val_imgs,   val_lbls   = images[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_imgs,  test_lbls  = images[n_train+n_val:],       labels[n_train+n_val:]

    # ---------- train-set stats (for class_weight & oversample) ----------
    classes_tr, counts_tr = np.unique(train_lbls, return_counts=True)
    class_counts_train = dict(zip(classes_tr.astype(int), counts_tr))
    weights = compute_class_weight(class_weight="balanced",
                                   classes=classes_tr,
                                   y=train_lbls)
    class_weights = dict(zip(classes_tr.astype(int), weights))

    # ---------- tf.data wrappers ----------
    train_ds_raw = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    val_ds_raw   = tf.data.Dataset.from_tensor_slices((val_imgs,   val_lbls))
    test_ds_raw  = tf.data.Dataset.from_tensor_slices((test_imgs,  test_lbls))

    return (train_ds_raw, val_ds_raw, test_ds_raw,
            class_weights, class_counts_train, class_counts_full)


def get_stratified_datasets(
    h5_path: str,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 123,
    shuffle: bool = True,
    shuffle_buffer: int = 1000
):
    """
    Load and stratify into train/val/test, then optionally shuffle each split.

    Returns
    -------
    train_ds_raw, val_ds_raw, test_ds_raw,
    class_weights, class_counts_train, class_counts_full
    """
    # 1) Load
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]
        labels = f["ans"][:]

    # 2) Full counts
    classes_all, counts_all = np.unique(labels, return_counts=True)
    class_counts_full = dict(zip(classes_all.astype(int), counts_all))

    # 3) First stratified split (test)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    trval_idx, test_idx = next(sss_test.split(images, labels))
    imgs_trval, lbls_trval = images[trval_idx], labels[trval_idx]
    imgs_test, lbls_test   = images[test_idx],   labels[test_idx]

    # 4) Second stratified split (val)
    val_frac_remain = val_split / (1.0 - test_split)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_remain, random_state=seed)
    train_idx, val_idx = next(sss_val.split(imgs_trval, lbls_trval))
    imgs_train = imgs_trval[train_idx]; lbls_train = lbls_trval[train_idx]
    imgs_val   = imgs_trval[val_idx];   lbls_val   = lbls_trval[val_idx]

    # 5) Train counts & weights
    classes_tr, counts_tr = np.unique(lbls_train, return_counts=True)
    class_counts_train = dict(zip(classes_tr.astype(int), counts_tr))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_tr,
        y=lbls_train
    )
    class_weights = dict(zip(classes_tr.astype(int), weights))

    # 6) Wrap in tf.data
    train_ds_raw = tf.data.Dataset.from_tensor_slices((imgs_train, lbls_train))
    val_ds_raw   = tf.data.Dataset.from_tensor_slices((imgs_val,   lbls_val))
    test_ds_raw  = tf.data.Dataset.from_tensor_slices((imgs_test,  lbls_test))

    # 7) Shuffle if requested
    if shuffle:
        train_ds_raw = train_ds_raw.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    return train_ds_raw, val_ds_raw, test_ds_raw, class_weights, class_counts_train, class_counts_full



# dataset preprocessing for the classification task, it takes the images to the format epected by tf's ResNet50
def preprocess(ds_raw,
               training=False,
               batch_size=32,
               target_size=(224,224)):
    """
    From a raw (image, label) dataset:
     - Optionally augment (if training=True)
     - Resize → Normalize for ResNet
     - Batch & prefetch
    Returns a fully preprocessed tf.data.Dataset.
    """
    # Define augmentation pipeline once
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(1.0),    # ±360°
        layers.RandomZoom(0.1),        # ±10%
        layers.RandomContrast(0.1),    # ±10%
        layers.GaussianNoise(0.01),    # σ=0.01
    ])

    def _map_fn(image, label):
        # 1) Augment only on training
        if training:
            image = data_augmentation(image)
        # 2) Resize to (224,224)
        image = tf.image.resize(image, target_size, method="bilinear")
        # 3) Normalize to ImageNet stats
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label

    ds = ds_raw.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        # Shuffle only the training set
        ds = ds.shuffle(buffer_size=1024, seed=123)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds



# dataset oversampling for the classification task, further information in the docstring
@tf.autograph.experimental.do_not_convert
def oversample_dataset(ds_raw,
                       class_counts: dict,
                       cutoff_count: int | None = None,
                       seed: int = 123):
    """Return **finite** balanced dataset using *cut-off* strategy.

    * Only classes with size < cutoff_count are upsampled.
    * cutoff_count default ⇒ second‑smallest class size (as requested).
    * The output has exactly  Σ max(orig_count, cutoff_count)  samples →
      model.fit can infer steps_per_epoch automatically ⇒ no infinite‑stream error.
    """
    # ---- pick default cut‑off (=2nd‑smallest class) ----
    if cutoff_count is None:
        sorted_counts = sorted(class_counts.values())
        if len(sorted_counts) < 2:
            raise ValueError("Need ≥2 classes to compute second‑smallest count.")
        cutoff_count = sorted_counts[1]

    per_class_ds = []        # list of finite per‑class datasets
    epoch_size   = 0         # running total for debug

    for cls, n_orig in sorted(class_counts.items()):
        # select this class
        ds_c = ds_raw.filter(lambda img, lbl, c=cls: tf.equal(lbl, c))
        ds_c = ds_c.shuffle(n_orig, seed=seed)

        if n_orig < cutoff_count:            # minority class → need repeats
            reps   = int(np.ceil(cutoff_count / n_orig))
            ds_c   = ds_c.repeat(reps).take(cutoff_count)
            n_final = cutoff_count
        else:                               # majority class → keep as‑is
            ds_c   = ds_c.take(n_orig)
            n_final = n_orig

        epoch_size += n_final
        per_class_ds.append(ds_c)

    # concatenate all finite streams & global shuffle
    balanced_ds = per_class_ds[0]
    for extra in per_class_ds[1:]:
        balanced_ds = balanced_ds.concatenate(extra)
    balanced_ds = balanced_ds.shuffle(epoch_size, seed=seed)

    # (optional) set cardinality for tf.data to know dataset length
    balanced_ds = balanced_ds.apply(tf.data.experimental.assert_cardinality(epoch_size))

    return balanced_ds



# preprocessing visualization for the classification task, takes a raw dataset and visualizes the preprocessing steps
def visualize_single_preprocessing(raw_ds_raw, class_names):
    """
    Pull one example from raw_ds_raw (shape [1,h,w,3]), then apply:
      1) RandomFlip
      2) RandomRotation
      3) RandomZoom
      4) RandomContrast
      5) GaussianNoise
      6) Resize to (224,224)
      7) ResNet50 caffe-style preprocess_input
    and plot after each.
    
    raw_ds_raw: a tf.data.Dataset yielding (image, label) in original 69×69 uint8
    class_names: dict mapping label→string
    """
    # --- 1) Grab one raw example ---
    img, lbl = next(iter(raw_ds_raw.batch(1)))
    img = tf.cast(img, tf.float32)  # [1,69,69,3]
    lbl = int(lbl.numpy()[0])
    
    # Layers for each step
    flip      = layers.RandomFlip("horizontal_and_vertical")
    rotate    = layers.RandomRotation(1.0)       # ±360°
    zoom      = layers.RandomZoom(0.1)           # ±10%
    contrast  = layers.RandomContrast(0.1)       # ±10%
    noise     = layers.GaussianNoise(0.01)       # σ=0.01
    resize_fn = lambda x: tf.image.resize(x, (224,224), method="bilinear")
    preprocess_fn = tf.keras.applications.resnet50.preprocess_input

    # Collect images after each step
    stages = [("Raw", img)]
    x = img
    x = flip(x);      stages.append(("Flip", x))
    x = rotate(x);    stages.append(("Rotate", x))
    x = zoom(x);      stages.append(("Zoom", x))
    x = contrast(x);  stages.append(("Contrast", x))
    x = noise(x);     stages.append(("Noise", x))
    x = resize_fn(x); stages.append(("Resize", x))
    x = preprocess_fn(x); stages.append(("Preproc", x))

    # --- Plotting ---
    n = len(stages)
    cols = 4
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(cols*3, rows*3))
    for i, (name, img_stage) in enumerate(stages):
        im = img_stage[0]  # remove batch dim
        # For display: if we've normalized (last stage), undo caffe
        if name == "Preproc":
            means_bgr = tf.constant([103.939,116.779,123.68], dtype=tf.float32)
            #im = im + means_bgr            # add back
            #im = im[..., ::-1]             # BGR→RGB
            #im = tf.clip_by_value(im/255., 0,1)
        else:
            # other stages still in [0–255] float
            im = tf.clip_by_value(im/255., 0,1)
        ax = plt.subplot(rows, cols, i+1)
        plt.imshow(im.numpy())
        plt.title(name)
        plt.axis("off")
    plt.suptitle(f"Label: {class_names[lbl]}", y=1.02)
    plt.tight_layout()
    plt.show()


#helper function to check if the model needs training or if it can load existing weights. used for every tensorflow model
def need_training(model: tf.keras.Model,
                  file_path: str,
                  force_retrain: bool = False):
    """
    Handle three cases, in order of priority:
      1) `force_retrain=True`     --> train from scratch
      2) full-model file exists   --> load it, skip training
      3) weights file exists      --> load weights into   `model`, skip training
      4) nothing found            --> must train
    Returns
    -------
    (model, bool)
        • model : the ready-to-use model
        • bool  : True  -> call model.fit
                   False -> skip training
    """
    if force_retrain:
        print("force_retrain=True → model will be trained from scratch.")
        return model, True

    # if the path ends with .keras or .h5 and contains a *full* model
    if file_path.endswith(".keras") and os.path.exists(file_path):
        print(f" Full model found at '{file_path}' → loading and skipping training.")
        model = tf.keras.models.load_model(file_path, compile=False)
        return model, False

    # weights-only fallback
    if os.path.exists(file_path):
        print(f" Weights found at '{file_path}' → loading and skipping training.")
        model.load_weights(file_path)
        return model, False

    print(f"No file at '{file_path}' → training required.")
    return model, True



# ============================= Grad CAM ================================


# ─── deprocess ─────────────────────────────────────────────────────────────
def resnet50_deprocess(x):
    """Invert tf.keras.applications.resnet50.preprocess_input."""
    # if it's a Tensor, pull it back into a NumPy array
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    # now we have a real ndarray and can copy/astype
    x = x.copy().astype(np.float32)
    # 1) add back ImageNet means
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 2) convert BGR → RGB
    x = x[..., ::-1]
    # 3) clip to [0,255]
    return np.clip(x, 0, 255).astype('uint8')


def _get_layer_recursive(m, name):
    for l in m.layers:
        if l.name == name:
            return l
        if isinstance(l, tf.keras.Model):
            try:
                return _get_layer_recursive(l, name)
            except ValueError:
                pass
    raise ValueError(f"No layer '{name}'. Top-level: {[l.name for l in m.layers]}")


def _find_last_conv(m):
    for l in reversed(m.layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            return l.name
        if isinstance(l, tf.keras.Model):
            try:
                return _find_last_conv(l)
            except ValueError:
                pass
    raise ValueError("No Conv2D found – pass last_conv_layer_name")

# ─── image canonicalisation ─────────────────────────────────────────────────

def _decode_if_bytes(img):
    return tf.io.decode_png(img, 3) if img.dtype == tf.string else img

def _reshape_if_flat(img):
    if tf.rank(img) == 2 and tf.shape(img)[-1] == 3:
        N = tf.shape(img)[0]
        side = tf.cast(tf.round(tf.sqrt(tf.cast(N, tf.float32))), tf.int32)
        if side * side == N:
            img = tf.reshape(img, (side, side, 3))
        else:
            img = tf.expand_dims(img, 0)
    return img

def _ensure_rgb(img):
    img = _decode_if_bytes(img)
    img = _reshape_if_flat(img)
    if tf.shape(img)[-1] == 1:
        img = tf.repeat(img, 3, -1)
    return tf.image.convert_image_dtype(img, tf.float32)

# ─── Grad-CAM core ──────────────────────────────────────────────────────────

def make_gradcam_heatmap(x, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv(model)
    layer = _get_layer_recursive(model, last_conv_layer_name)
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        pred_index = tf.argmax(predictions[0]) if pred_index is None else pred_index
        loss = tf.squeeze(predictions[:, pred_index])
    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_outputs[0] @ pooled[..., None])
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + tf.keras.backend.epsilon())
    return heat.numpy()

# ─── visuals ─────────────────────────────────────────────────────────────────
'''
# old version, kept beacause it works but the contrast is very low and the gradcam is pretty hard to see. 
def _to_display(img):
    arr = img.numpy() if isinstance(img, tf.Tensor) else img
    return np.clip(arr.astype('float32')/255. if arr.dtype != np.float32 else arr, 0, 1)

def overlay_heatmap(hm, img, alpha=.4):
    base = _to_display(img)
    jet = plt.get_cmap('jet')(np.arange(256))[:, :3]
    jet = tf.image.resize(jet[np.uint8(255*hm)], base.shape[:2]).numpy()
    return np.clip(base*(1-alpha) + jet*alpha, 0, 1)
'''

def smart_contrast(img, mask=None, clip=0.02):
    """
    img  : float32 RGB [0,1]
    mask : optional binary mask (True where galaxy is)
    CLAHE only inside mask; elsewhere apply mild gamma.
    """
    out = img.copy()
    if mask is None:
        # global gamma 0.8 to brighten
        out = np.power(out, 0.8)
    else:
        # clahe inside mask
        for c in range(3):
            ch = out[..., c]
            ch_eq = exposure.equalize_adapthist(ch, clip_limit=clip)
            out[..., c] = np.where(mask, ch_eq, np.power(ch, 0.9))
    return np.clip(out, 0, 1)

def overlay_heatmap_bad(hm, img_raw, alpha=0.6, top_p=0.20, blur=3):
    """
    hm      : 2-D heat-map [0,1] (Grad-CAM).
    img_raw : uint8 or float32 RGB, uint8 0-255 expected.
    """
    # --- prep image -------------------------------------------------
    img = img_raw.astype("float32")
    if img.max() > 1.01:
        img /= 255.0
    # optional galaxy mask: pixels brighter than median
    mask = img.mean(-1) > np.median(img.mean(-1))
    base = smart_contrast(img, mask)

    # --- prep heat-map ---------------------------------------------
    hm = gaussian(hm, sigma=blur, preserve_range=True)           # smooth
    thresh = np.quantile(hm, 1-top_p)                  # keep top P %
    hm = np.clip((hm-thresh)/(hm.max()-thresh+1e-8),0,1)

    cmap = plt.get_cmap("magma")
    rgba = cmap(hm)                                    # (H,W,4)
    rgba[..., -1] *= alpha                             # scale alpha

    # resize if needed
    if hm.shape[:2] != base.shape[:2]:
        rgba = tf.image.resize(rgba, base.shape[:2]).numpy()

    # --- alpha-blend -----------------------------------------------
    out = base*(1-rgba[...,3:]) + rgba[..., :3]*rgba[...,3:]
    return np.clip(out, 0, 1)


# standard overlay function
'''
def overlay_heatmap(hm, img, alpha=0.45, cmap="turbo"):
    """
    hm  : 2-D Grad-CAM heat-map in [0,1]   (any H×W)
    img : RGB uint8/float32 image          (H×W×3)
    """
    # ─── prep background ─────────────────────────────────────────────
    base = img.astype("float32")
    if base.max() > 1.01:          # uint8 → scale
        base /= 255.0

    # ─── normalise & resize heat-map ─────────────────────────────────
    hm = np.maximum(hm, 0)
    hm = hm / (hm.max() + 1e-8)
    if hm.shape != base.shape[:2]:
        hm = tf.image.resize(hm[..., None], base.shape[:2]).numpy()[..., 0]

    # ─── colourise ───────────────────────────────────────────────────
    color = plt.get_cmap(cmap)(hm)[..., :3]      # RGB, same H×W
    weight = (alpha * hm)[..., None]             # per-pixel alpha

    # ─── blend ───────────────────────────────────────────────────────
    out = base * (1 - weight) + color * weight
    return np.clip(out, 0, 1)
'''

def overlay_heatmap(
    hm,
    img,
    alpha: float = 0.60,      # final opacity scale
    *,
    cmap: str = "magma",      # matplotlib colormap name
    top_p: float = 0.4,      # fraction of pixels to keep (0.0–1.0)
    sigma: float = 4.0,       # blur radius (in pixels)
    gamma: float = 0.8        # gamma-correct the heatmap
):
    """
    hm   : 2-D array ∈ [0,1] (Grad-CAM output)
    img  : RGB uint8 or float32 [0..255]/[0..1]
    """

    # 1) Prepare your background image (no contrast tweak here)
    base = img.astype("float32")
    if base.max() > 1.01:
        base /= 255.0

    # 2) Normalise + gamma-stretch the raw GradCAM
    hm = np.maximum(hm, 0)
    hm = hm / (hm.max() + 1e-8)
    hm = hm ** gamma             # <-- gamma here

    # 3) Resize to image shape if needed
    if hm.shape != base.shape[:2]:
        hm = tf.image.resize(hm[..., None], base.shape[:2]).numpy()[..., 0]

    # 4) Sparsify to only the top_p fraction
    thresh = np.quantile(hm, 1 - top_p)
    hm = np.where(hm >= thresh, hm, 0.0)

    # 5) Smooth the mask
    hm = gaussian(hm, sigma=sigma, preserve_range=True)
    hm = hm / (hm.max() + 1e-8)

    # 6) Turn it into RGB via your colormap
    color = plt.get_cmap(cmap)(hm)[..., :3]

    # 7) Blend: higher-hm pixels get more of “color”, lower-hm more of “base”
    w = (alpha * hm)[..., None]
    out = base * (1 - w) + color * w

    return np.clip(out, 0, 1)


# ─── sample collection ───────────────────────────────────────────────────────

def collect_samples(ds, model, preproc=None):
    """Gather first TP+FN per class. Always scans entire dataset to capture correct hits."""
    h, w = model.input_shape[1:3]
    n = model.output_shape[-1]
    bucket = {i: {'correct': None, 'incorrect': None} for i in range(n)}

    for xs, ys in ds:
        if tf.rank(xs) == 3:
            xs, ys = tf.expand_dims(xs, 0), tf.expand_dims(ys, 0)
        xs = tf.map_fn(_ensure_rgb, xs, fn_output_signature=tf.float32)
        xs_resized = tf.image.resize(xs, (h, w))
        xs_model = preproc(xs_resized) if preproc else xs_resized
        preds = model.predict(xs_model, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = ys.numpy() if isinstance(ys, tf.Tensor) else ys

        for raw_img, t, p in zip(xs, y_true, y_pred):
            tag = 'correct' if t == p else 'incorrect'
            if bucket[t][tag] is None:
                bucket[t][tag] = (raw_img, p)
    return bucket

# ─── public API ─────────────────────────────────────────────────────────────

def plot_gradcam_matrix(
    model,
    dataset,
    class_names=None,
    last_conv_layer_name=None,
    preprocess_fn=None,
    deprocess_fn=None,
    classes_to_plot: list[int] | None = None,
    alpha=.4,
    figsize=(8, 12)
):
    """
    Plot Grad-CAM overlays for specified classes, but only apply deprocess_fn
    to the *preprocessed* input, not to your raw uint8 images.
    """
    h, w = model.input_shape[1:3]
    n_classes = model.output_shape[-1]

    # 1) first scan the dataset and bucket one TP & one FN per class
    bucket = {c: {"correct": None, "incorrect": None} for c in range(n_classes)}
    for xs, ys in dataset:
        if tf.rank(xs) == 3:
            xs, ys = tf.expand_dims(xs, 0), tf.expand_dims(ys, 0)

        # xs_raw: uint8 RGB [0..255]
        xs_raw = xs

        # Build the model-input:
        xs_resized = tf.image.resize(xs_raw, (h, w), method="bilinear")
        if preprocess_fn:
            xs_proc = preprocess_fn(tf.cast(xs_resized, tf.float32))
        else:
            # if no preprocess, assume xs_resized is already correct dtype
            xs_proc = xs_resized

        preds = model.predict(xs_proc, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = ys.numpy()

        for raw_img, proc_img, t, p in zip(xs_raw, xs_proc, y_true, y_pred):
            tag = "correct" if t == p else "incorrect"
            if bucket[t][tag] is None:
                bucket[t][tag] = (raw_img, proc_img, int(p))
        
        if all(
            bucket[c]["correct"] is not None and bucket[c]["incorrect"] is not None
            for c in bucket if (classes_to_plot is None or c in classes_to_plot)
        ):
            break

    # 2) filter by classes_to_plot and existence
    filtered = {
        c: rec for c, rec in bucket.items()
        if (rec["correct"] or rec["incorrect"]) and
           (classes_to_plot is None or c in classes_to_plot)
    }
    if not filtered:
        raise ValueError("No usable samples found for given classes_to_plot")

    class_names = class_names or [str(i) for i in range(n_classes)]
    rows = len(filtered)
    fig, axes = plt.subplots(rows, 2, figsize=figsize)

    # 3) plot each cell
    for row, (cls, rec) in enumerate(filtered.items()):
        for col, tag in enumerate(["correct", "incorrect"]):
            ax = axes[row, col] if rows>1 else axes[col]
            if rec[tag] is None:
                ax.axis("off")
                ax.set_title(f"No {tag}", color="gray", fontsize=9)
                continue

            raw_img, proc_img, pred = rec[tag]

            # a) display image: if you gave a deprocess_fn, apply it to the *pre*-processed
            #    tensor (proc_img), otherwise just normalize the raw uint8
            if deprocess_fn:
                # proc_img is shape (h,w,3) float32 BGR-mean subtracted
                disp = deprocess_fn(proc_img.numpy())
            else:
                # normalize raw to [0,1]
                disp = np.clip(raw_img.numpy().astype("float32")/255.0, 0, 1)

            # b) compute heatmap
            #    make sure we feed a batch of size 1 into it:
            inp = proc_img[None]  # shape (1,h,w,3)
            hm = make_gradcam_heatmap(inp, model, last_conv_layer_name, pred)

            # c) overlay & title
            ax.imshow(overlay_heatmap(hm, disp, alpha))
            ax.axis("off")
            title = (
                #f"{'✓' if tag=='correct' else '✗'} "
                f"True={class_names[cls]}"
                + (f" | Pred={class_names[pred]}" if tag=="incorrect" else "")
            )
            ax.set_title(title, fontsize=9)

    fig.suptitle("Grad-CAM per class", fontsize=12)
    fig.tight_layout()
    return fig

'''
def plot_gradcam_means(
    model,
    dataset,
    class_ids,
    max_per_bucket=30,
    last_conv="conv5_block3_out",
    preprocess=tf.keras.applications.resnet50.preprocess_input,
    cmap="magma",
    figsize=None
):
    """
    • Scans `dataset` until it collects `max_per_bucket` correctly- and
      mis-classified samples *per class* in `class_ids`.
    • Averages the Grad-CAMs within each bucket.
    • Plots mean(correct), mean(incorrect), and difference for each class.
    """
    if figsize is None:
        figsize = (len(class_ids)*3, 4)

    h, w = model.input_shape[1:3]
    buckets = {
        cid: {"ok": [], "bad": []}
        for cid in class_ids
    }

    # -------- accumulate heat-maps -------------
    for xs, ys in dataset.batch(32):
        xs_res = tf.image.resize(xs, (h, w))
        xs_in  = preprocess(tf.cast(xs_res, tf.float32))
        preds  = model.predict(xs_in, verbose=0).argmax(1)
        for img_in, y_true, y_pred in zip(xs_in, ys.numpy(), preds):
            if y_true not in buckets:              # class not requested
                continue
            tag = "ok" if y_true == y_pred else "bad"
            if len(buckets[y_true][tag]) >= max_per_bucket:
                continue
            hm = make_gradcam_heatmap(img_in[None], model, last_conv, y_pred)
            buckets[y_true][tag].append(hm)
        # early exit: all buckets full?
        if all(
            len(b["ok"]) >= max_per_bucket and len(b["bad"]) >= max_per_bucket
            for b in buckets.values()
        ):
            break

    # -------- plot averages --------------------
    n = len(class_ids)
    fig, axes = plt.subplots(3, n, figsize=figsize, dpi=120,
                             sharex=True, sharey=True)
    for col, cid in enumerate(class_ids):
        ok  = np.mean(buckets[cid]["ok"],  axis=0)
        bad = np.mean(buckets[cid]["bad"], axis=0)
        diff = ok - bad
        for row, mat, title in zip(
            [0, 1, 2],
            [ok, bad, diff],
            ["Mean OK", "Mean Err", "OK − Err"]
        ):
            ax = axes[row, col]
            ax.imshow(mat, cmap=cmap); ax.axis("off")
            if row == 0:
                ax.set_title(f"Class {cid}", fontsize=10)
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
    fig.suptitle("Averaged Grad-CAMs ({} per bucket)".format(max_per_bucket))
    fig.tight_layout()
    return fig
'''

def plot_gradcam_means(
    model,
    dataset,
    class_ids,
    max_per_bucket: int = 30,
    last_conv: str = "conv5_block3_out",
    preprocess = tf.keras.applications.resnet50.preprocess_input,
    cmap: str = "magma",
    upsample_to: int = 40,       # <-- new: target resolution for display
    figsize: tuple | None = None
):
    """
    • Scan `dataset` until you collect `max_per_bucket` correctly- and
      mis-classified heat-maps for each class in `class_ids`.
    • Compute mean(correct), mean(incorrect), and their difference.
    • Upsample each mean map to `upsample_toxupsample_to` for display.
    • Plot a 3xN grid with titles on every cell.
    """
    if figsize is None:
        figsize = (len(class_ids)*6, 10)

    # 1) collect
    h, w = model.input_shape[1:3]
    buckets = {cid: {"ok": [], "bad": []} for cid in class_ids}

    for xs, ys in dataset.batch(32):
        xs_res = tf.image.resize(xs, (h, w))
        xs_in  = preprocess(tf.cast(xs_res, tf.float32))
        preds  = model.predict(xs_in, verbose=0).argmax(1)

        for img_in, y_true, y_pred in zip(xs_in, ys.numpy(), preds):
            if y_true not in buckets:
                continue
            tag = "ok" if y_true == y_pred else "bad"
            if len(buckets[y_true][tag]) < max_per_bucket:
                hm = make_gradcam_heatmap(
                    img_in[None], model, last_conv, pred_index=y_pred
                )
                buckets[y_true][tag].append(hm)

        # early stopping when all full
        if all(
            len(b["ok"])>=max_per_bucket and len(b["bad"])>=max_per_bucket
            for b in buckets.values()
        ):
            break

    # 2) plot
    n = len(class_ids)
    fig, axes = plt.subplots(
        3, n,
        figsize=figsize,
        dpi=120,
        # sharex=True, sharey=True  ← remove if you want independent axes
    )
    row_labels = ["Mean OK", "Mean Err", "OK − Err"]

    for col, cid in enumerate(class_ids):
        ok  = np.mean(buckets[cid]["ok"],  axis=0)
        bad = np.mean(buckets[cid]["bad"], axis=0)
        diff= ok - bad

        for row, mat, label in zip(range(3), [ok, bad, diff], row_labels):
            ax = axes[row, col]

            # upsample to (upsample_to, upsample_to)
            mat_up = tf.image.resize(
                mat[...,None],
                (upsample_to, upsample_to),
                method="bilinear"
            ).numpy()[...,0]

            im = ax.imshow(mat_up, cmap=cmap, aspect="equal")
            ax.axis("off")

            # title every subplot
            ax.set_title(f"{label}\nClass {cid}", fontsize=9)

    fig.suptitle(f"Averaged Grad-CAMs ({max_per_bucket} per bucket)", fontsize=12)
    fig.tight_layout()
    return fig

## ========================= DNN preprocess =========================

def compute_dataset_mean_std(ds):
    """
    Computes per-channel mean and std for a tf.data.Dataset yielding (image, label),
    where image is a uint8 or float Tensor of shape (H, W, C).
    Returns two numpy arrays of shape (3,): mean and std of pixel values in [0,1].
    """
    sum_ = np.zeros(3, dtype=np.float64)
    sum_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for image, label in ds:
        img = image.numpy().astype(np.float32) / 255.0
        # shape (H, W, 3)
        sum_ += img.sum(axis=(0,1))
        sum_sq += (img ** 2).sum(axis=(0,1))
        count += img.shape[0] * img.shape[1]
    mean = sum_ / count
    var = sum_sq / count - mean**2
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)

def preprocess_no_resize_with_stats(
    ds_raw,
    mean,
    std,
    training=False,
    batch_size=32
):
    """
    Preprocess 69x69 images with dataset-specific normalization.
    Args:
      ds_raw: tf.data.Dataset yielding (image, label), image shape (69,69,3), dtype uint8 or float.
      mean, std: numpy arrays shape (3,) for channel-wise mean and std in [0,1] domain.
    Returns:
      preprocessed tf.data.Dataset yielding (normalized_image, label).
    """
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(1.0),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.GaussianNoise(0.01),
    ])

    # convert mean and std to tensors of shape (1,1,3)
    mean_tf = tf.constant(mean.reshape((1,1,3)), dtype=tf.float32)
    std_tf = tf.constant(std.reshape((1,1,3)), dtype=tf.float32)

    def _map_fn(image, label):
        if training:
            image = aug(image)
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - mean_tf) / std_tf
        return image, label

    ds = ds_raw.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1024, seed=123)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

## __________________ PCA __________________________

def fit_incremental_pca(train_ds, n_components: int, batch_size: int = 256):
    """
    Fits an IncrementalPCA so that the first partial_fit call
    sees at least n_components samples.
    """
    # 1) Unbatch and re-batch so first batch >= n_components
    ds_unbatched = train_ds.unbatch().map(
        lambda img, lbl: img,  # drop labels here
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # first batch
    first = ds_unbatched.batch(n_components).take(1)
    
    ipca = IncrementalPCA(n_components=n_components)
    for batch in first:
        flat = batch.numpy().reshape(batch.shape[0], -1)
        ipca.partial_fit(flat)
    
    # 2) Now feed the rest in batch_size chunks
    rest = ds_unbatched.skip(n_components).batch(batch_size)
    for batch in rest:
        flat = batch.numpy().reshape(batch.shape[0], -1)
        ipca.partial_fit(flat)
    
    return ipca

def transform_dataset_with_pca(ds, pca, batch_size: int = 32):
    """
    Transforms a tf.data.Dataset through a fitted PCA and returns a new dataset.
    """
    X_list, y_list = [], []
    for images, labels in ds:
        flat = images.numpy().reshape(images.shape[0], -1)
        X_pca = pca.transform(flat)
        X_list.append(X_pca)
        y_list.append(labels.numpy())
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    ds_pca = tf.data.Dataset.from_tensor_slices((X, y))
    if ds.element_spec[0].shape[0] is None:  # check if shuffling needed
        ds_pca = ds_pca.shuffle(buffer_size=X.shape[0], seed=123)
    return ds_pca.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# similar to need_training, but for PCA models
def pca_need_fit(path: str, force: bool, pca):
    """
    Either loads an existing PCA model or signals that fitting is required.

    Parameters
    ----------
    path : str
        Filesystem path to save/load the PCA (e.g. "pca512.joblib").
    force : bool
        If True, ignore any existing file and retrain.
    pca : sklearn.decomposition.PCA
        An unfitted PCA instance (with desired n_components).

    Returns
    -------
    need_train : bool
        • True  → PCA needs to be fit (and then you should joblib.dump it).  
        • False → PCA was loaded from disk; no fitting needed.  
    pca_model : PCA
        The loaded (or to-be-fitted) PCA instance.
    """
    if force:
        print("force=True → PCA will be trained from scratch.")
        return True, pca

    if not os.path.exists(path):
        print(f"🔎 No PCA file at '{path}' → training required.")
        return True, pca

    # file exists and force is False → load it
    print(f"PCA found at '{path}' → loading and skipping training.")
    loaded = joblib.load(path)
    return False, loaded


def get_or_cache_pca_dataset(raw_ds,
                             pca,
                             cache_path: str,
                             batch_size: int = 32,
                             force: bool = False,
                             shuffle_cache=True):
    """
    Either loads a cached PCA-transformed tf.data.Dataset from `cache_path`,
    or runs `transform_dataset_with_pca`, saves it, and returns it.

    Parameters
    ----------
    raw_ds : tf.data.Dataset
        Yields (image, label) pairs; images are 69×69×3 floats.
    pca : fitted sklearn PCA
        Used to transform flattened images → n_components.
    cache_path : str
        Directory path to save/load the cached dataset.
    batch_size : int
    force : bool
        If True, ignore any existing cache and recompute.

    Returns
    -------
    ds : tf.data.Dataset
        Yields (X_pca, label) batches.
    """
    # 1) If cache exists and not forcing, load it
    if not force and tf.io.gfile.exists(cache_path):
        print(f"Loading PCA-features from '{cache_path}'")
        ds = tf.data.experimental.load(
            cache_path,
            element_spec=(
                tf.TensorSpec(shape=(None, pca.n_components_), dtype=tf.float32),
                tf.TensorSpec(shape=(None,),             dtype=tf.int32)
            )
        )
        return ds

    # 2) Otherwise, build it once
    print(f"Cache miss or force=True → computing and saving to '{cache_path}'")
    X_parts, y_parts = [], []
    for images, labels in raw_ds:
        flat   = tf.reshape(images, [tf.shape(images)[0], -1]).numpy()
        X_pca  = pca.transform(flat)
        X_parts.append(X_pca.astype(np.float32))
        y_parts.append(labels.numpy().astype(np.int32))

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    ds = (tf.data.Dataset
            .from_tensor_slices((X_all, y_all))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

    if shuffle_cache:
        ds = ds.shuffle(buffer_size=X_all.shape[0], seed=123)
    # 3) Save to cache_path
    tf.data.experimental.save(ds, cache_path)
    return ds

## ========================= anomaly detection =========================

def need_anom_datasets(cache_dir: str,
                       force_recompute: bool = False,
                       batch_size: int       = 128,
                       seed: int             = 123):
    """
    Returns 9‐tuple:
      recompute_flag,
      embs_train, embs_val, embs_test, embs_anom,
      ds_train,    ds_val,    ds_test,    ds_anom

    If recompute_flag==False, all embs_* are loaded from .npy
    and all ds_* from tf.data.experimental.load(). Otherwise
    they’re all None and you should recompute & then call save_*.
    """
    # paths for embeddings
    embs_dir = cache_dir + "/embeddings"
    arr_paths = {k: os.path.join(embs_dir, f"{k}.npy")
                 for k in ("embs_train","embs_val","embs_test","embs_anom")}
    # paths for tf.data
    ds_paths  = {k: os.path.join(cache_dir, k.replace("embs","ds"))
                 for k in ("embs_train","embs_val","embs_test","embs_anom")}

    # do we need to rebuild?
    if force_recompute or not (
       all(os.path.exists(p) for p in arr_paths.values()) and
       all(os.path.isdir(p) for p in ds_paths.values())
    ):
        if force_recompute:
            print("force_recompute=True → will rebuild embeddings + datasets")
        else:
            print(f"Cache missing/incomplete in '{cache_dir}' → rebuild required")
        return True, *(None,)*8

    # load embeddings
    embs_train = np.load(arr_paths["embs_train"])
    embs_val   = np.load(arr_paths["embs_val"])
    embs_test  = np.load(arr_paths["embs_test"])
    embs_anom  = np.load(arr_paths["embs_anom"])
    print(f"Loaded embeddings from '{cache_dir}'")

    # load datasets
    spec = tf.TensorSpec(shape=(None, embs_train.shape[1]), dtype=tf.float32)
    ds_train = tf.data.experimental.load(ds_paths["embs_train"], spec)
    ds_val   = tf.data.experimental.load(ds_paths["embs_val"],   spec)
    ds_test  = tf.data.experimental.load(ds_paths["embs_test"],  spec)
    ds_anom  = tf.data.experimental.load(ds_paths["embs_anom"],  spec)
    print(f"Loaded datasets   from '{cache_dir}'")

    return False, embs_train, embs_val, embs_test, embs_anom, ds_train, ds_val, ds_test, ds_anom

def save_anom_cache(cache_dir: str,
                    embs_train=None, embs_val=None,
                    embs_test=None,  embs_anom=None,
                    ds_train=None,   ds_val=None,
                    ds_test=None,    ds_anom=None):
    """
    Save embeddings (NumPy arrays) or datasets (tf.data.Dataset) depending on what you pass.
    - If embs_* is a NumPy array, saves to <cache_dir>/<name>.npy
    - If ds_*   is a tf.data.Dataset, saves to <cache_dir>/<name>/
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Embeddings → .npy
    for name, arr in [
        ("embs_train", embs_train), ("embs_val",   embs_val),
        ("embs_test",  embs_test),  ("embs_anom", embs_anom)
    ]:
        if arr is not None:
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"{name!r} passed but is not a NumPy array")
            path = os.path.join(cache_dir, f"{name}.npy")
            np.save(path, arr)
            print(f"Saved embeddings {name} → {path}")

    # Datasets → tf.data.experimental.save
    for name, ds in [
        ("ds_train", ds_train), ("ds_val",   ds_val),
        ("ds_test",  ds_test),  ("ds_anom", ds_anom)
    ]:
        if ds is not None:
            if not isinstance(ds, tf.data.Dataset):
                raise ValueError(f"{name!r} passed but is not a tf.data.Dataset")
            path = os.path.join(cache_dir, name)
            # overwrite if exists
            if os.path.isdir(path):
                tf.io.gfile.rmtree(path)
            tf.data.experimental.save(ds, path)
            print(f"Saved dataset    {name} → {path}")

def get_anomaly_datasets(h5_path,
                         normal_classes=(0,1,2),
                         anomaly_classes=None,
                         n_anomalies=100,
                         val_split: float = 0.2,
                         test_split: float = 0.05,
                         seed: int = 123):
    """
    Prepare datasets for anomaly detection:
      - Train on "normal" classes only (stratified into train/val/test_normal)
      - Test anomalies separately

    Parameters
    ----------
    h5_path : str
        Path to HDF5 file containing 'images' and 'ans' datasets.
    normal_classes : tuple of int
        Class labels to treat as in-distribution (normal).
    anomaly_classes : list of int, optional
        Class labels to sample as anomalies. Defaults to last three classes.
    n_anomalies : int
        Number of anomaly samples to draw (total across all anomaly_classes).
    val_split : float
        Fraction of normal data to reserve for validation.
    test_split : float
        Fraction of normal data to reserve as a hold-out test set (good galaxies).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_ds_raw : tf.data.Dataset
        Dataset of (image, label) for training (normal labels).
    val_ds_raw : tf.data.Dataset
        Dataset of (image, label) for validation (normal labels).
    test_ds_raw : tf.data.Dataset
        Dataset of (image, label) for normal test (good galaxies).
    anomaly_ds_raw : tf.data.Dataset
        Dataset of (image, is_anomaly) for anomaly testing (1=anomaly).
    class_weights : dict
        Class weights for training normals.
    class_counts_train : dict
        Counts of each normal class in the training split.
    class_counts_full : dict
        Counts of each class in the full dataset.
    anomaly_counts_test : dict
        Counts of anomalies sampled per anomaly class.
    """
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    # Load data
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]  # (N,H,W,C)
        labels = f["ans"][:]

    # Full dataset counts
    classes_all, counts_all = np.unique(labels, return_counts=True)
    class_counts_full = dict(zip(classes_all.astype(int), counts_all))

    # Default anomaly classes: last 3 unique classes
    if anomaly_classes is None:
        sorted_classes = sorted(classes_all.astype(int))
        anomaly_classes = sorted_classes[-3:]

    rng = np.random.default_rng(seed)

    # Extract normal data
    normal_mask = np.isin(labels, normal_classes)
    normal_imgs = images[normal_mask]
    normal_lbls = labels[normal_mask]

    # Shuffle normals
    idx_norm = rng.permutation(len(normal_lbls))
    normal_imgs, normal_lbls = normal_imgs[idx_norm], normal_lbls[idx_norm]

    # Split train / val / test_normal for normals
    n_norm = len(normal_lbls)
    n_val = int(n_norm * val_split)
    n_test_norm = int(n_norm * test_split)
    n_train = n_norm - n_val - n_test_norm

    train_imgs = normal_imgs[:n_train]
    train_lbls = normal_lbls[:n_train]
    val_imgs   = normal_imgs[n_train:n_train+n_val]
    val_lbls   = normal_lbls[n_train:n_train+n_val]
    test_norm_imgs = normal_imgs[n_train+n_val: n_train+n_val+n_test_norm]
    test_norm_lbls = normal_lbls[n_train+n_val: n_train+n_val+n_test_norm]

    # Compute train class counts
    classes_tr, counts_tr = np.unique(train_lbls, return_counts=True)
    class_counts_train = dict(zip(classes_tr.astype(int), counts_tr))

    # Compute class weights for training normals
    weights = compute_class_weight(class_weight="balanced",
                                   classes=classes_tr,
                                   y=train_lbls)
    class_weights = dict(zip(classes_tr.astype(int), weights))

    # Build anomaly test set
    n_per_class = int(np.ceil(n_anomalies / len(anomaly_classes)))
    anomaly_imgs_list, anomaly_labels_list = [], []
    anomaly_counts_test = {}
    for c in anomaly_classes:
        c_mask = (labels == c)
        imgs_c = images[c_mask]
        n_take = min(n_per_class, len(imgs_c))
        idx_c = rng.choice(len(imgs_c), size=n_take, replace=False)
        anomaly_imgs_list.append(imgs_c[idx_c])
        anomaly_labels_list.extend([1] * n_take)
        anomaly_counts_test[c] = n_take

    anomaly_images = np.concatenate(anomaly_imgs_list, axis=0)
    anomaly_labels = np.array(anomaly_labels_list, dtype=np.int32)

    # Shuffle anomaly set
    idx_ano = rng.permutation(len(anomaly_labels))
    anomaly_images, anomaly_labels = anomaly_images[idx_ano], anomaly_labels[idx_ano]

    # Create tf.data datasets
    train_ds_raw = tf.data.Dataset.from_tensor_slices((train_imgs, train_lbls))
    val_ds_raw   = tf.data.Dataset.from_tensor_slices((val_imgs,   val_lbls))
    test_ds_raw  = tf.data.Dataset.from_tensor_slices((test_norm_imgs, test_norm_lbls))
    anomaly_ds_raw = tf.data.Dataset.from_tensor_slices((anomaly_images, anomaly_labels))

    return (
        train_ds_raw,
        val_ds_raw,
        test_ds_raw,
        anomaly_ds_raw,
        class_weights,
        class_counts_train,
        class_counts_full,
        anomaly_counts_test
    )


def get_stratified_anomaly_datasets(
    h5_path,
    normal_classes=(0,1,2),
    anomaly_classes=None,
    n_anomalies=100,
    val_split=0.2,
    test_split=0.05,
    seed=123,
    shuffle=True,
    shuffle_buffer=1000
):
    """
    Stratified normals + sampled anomalies, with optional shuffle.

    Returns:
      train_ds_raw, val_ds_raw, test_ds_raw, anomaly_ds_raw,
      class_weights, class_counts_train, class_counts_full, anomaly_counts_test
    """
    # 1) Load all images + labels
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]  
        labels = f["ans"][:]     

    # 2) Full-dataset class counts
    classes_all, counts_all = np.unique(labels, return_counts=True)
    class_counts_full = dict(zip(classes_all.astype(int), counts_all))

    # 3) Determine anomaly classes if needed
    if anomaly_classes is None:
        sorted_cls = sorted(classes_all.astype(int))
        anomaly_classes = sorted_cls[-3:]

    # 4) Filter normals
    mask_norm  = np.isin(labels, normal_classes)
    imgs_norm   = images[mask_norm]
    lbls_norm   = labels[mask_norm]

    # 5a) Stratify-out test normals
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    trval_idx, test_norm_idx = next(sss1.split(imgs_norm, lbls_norm))
    imgs_trval      = imgs_norm[trval_idx]
    lbls_trval      = lbls_norm[trval_idx]
    imgs_test_norm  = imgs_norm[test_norm_idx]
    lbls_test_norm  = lbls_norm[test_norm_idx]

    # 5b) Stratify train vs. val from remaining normals
    val_frac_remain = val_split / (1.0 - test_split)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_remain, random_state=seed)
    train_idx, val_idx = next(sss2.split(imgs_trval, lbls_trval))
    imgs_train = imgs_trval[train_idx]; lbls_train = lbls_trval[train_idx]
    imgs_val   = imgs_trval[val_idx];   lbls_val   = lbls_trval[val_idx]

    # 6) Train-class counts & weights
    cls_tr, cnt_tr = np.unique(lbls_train, return_counts=True)
    class_counts_train = dict(zip(cls_tr.astype(int), cnt_tr))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=cls_tr,
        y=lbls_train
    )
    class_weights = dict(zip(cls_tr.astype(int), weights))

    # 7) Sample anomalies
    rng = np.random.default_rng(seed)
    per_class = int(np.ceil(n_anomalies / len(anomaly_classes)))
    imgs_ano_list = []; anomaly_counts_test = {}
    for c in anomaly_classes:
        imgs_c = images[labels == c]
        take   = min(per_class, len(imgs_c))
        idx_c  = rng.choice(len(imgs_c), size=take, replace=False)
        imgs_ano_list.append(imgs_c[idx_c])
        anomaly_counts_test[c] = take

    imgs_anomaly = np.concatenate(imgs_ano_list, axis=0)
    lbls_anomaly = np.ones(len(imgs_anomaly), dtype=np.int32)

    # 8) Wrap into tf.data.Dataset
    train_ds_raw   = tf.data.Dataset.from_tensor_slices((imgs_train,     lbls_train))
    val_ds_raw     = tf.data.Dataset.from_tensor_slices((imgs_val,       lbls_val))
    test_ds_raw    = tf.data.Dataset.from_tensor_slices((imgs_test_norm, lbls_test_norm))
    anomaly_ds_raw = tf.data.Dataset.from_tensor_slices((imgs_anomaly,   lbls_anomaly))

    # 9) Shuffle if requested (only training needs full reshuffle each epoch)
    if shuffle:
        train_ds_raw   = train_ds_raw.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    return (
        train_ds_raw,
        val_ds_raw,
        test_ds_raw,
        anomaly_ds_raw,
        class_weights,
        class_counts_train,
        class_counts_full,
        anomaly_counts_test
    )


## __________________________ GMM __________________________
#function to check if the GMM model needs to be trained or if it can load existing weights
def gmm_need_fit(path, force, gmm):
    if force:
        print(f"force=True → GMM will be trained from scratch.")
        return True, gmm
    if not os.path.exists(path):
        print(f"🔎 No GMM file at '{path}' → training required.")
        return True, gmm
    if os.path.exists(path):
        print(f" GMM found at '{path}' → loading and skipping training.")
        trained_gmm = joblib.load("gmm_model.joblib")
        return False, trained_gmm


# 3)helper for anomaly scoring
# ---------------------------------------
def compute_GMM_anomaly_scores(dataset_raw,feature_extractor, gmm):
    """
    Given a raw (image, label) dataset, returns:
      - scores: 1D array of anomaly scores
      - labels: original labels (0=normal class index or 1=anomaly)
    """
    # Preprocess (no augmentation)
    ds = preprocess(
        dataset_raw,
        training=False,
        batch_size=BATCH_SIZE,
        target_size=TARGET_SIZE
    )
    ## Extract embeddings
    Z = feature_extractor.predict(ds)             # (N,32)
    ## Compute negative log-likelihood
    log_probs = gmm.score_samples(Z)              # (N,)
    #log_probs = gmm.score_samples(dataset_raw)
    scores = -log_probs                           # higher → more anomalous

    # Retrieve the true labels from the raw dataset
    labels = np.concatenate([y.numpy() for x, y in dataset_raw.batch(BATCH_SIZE)])
    return scores, labels



def vae_anomaly_score(model, ds, beta=0.5):
    """
    Compute anomaly scores over a tf.data.Dataset of shape [batch, D].
    Returns a flat NumPy array of length = total samples in ds.
    """
    scores = []
    for batch in ds:
        # ensure float32 tensor
        x = tf.cast(batch, tf.float32)
        # encode
        z_mean, z_logvar = model.encode(x)
        # use mean only
        z = z_mean
        # decode
        x_recon = model.decode(z)
        # move to NumPy
        x_np       = x.numpy()
        x_recon_np = x_recon.numpy()
        z_mean_np  = z_mean.numpy()
        z_log_np   = z_logvar.numpy()
        # compute losses
        recon_err = np.sum((x_np - x_recon_np)**2, axis=1)
        kl_term   = -0.5 * np.sum(1 + z_log_np - z_mean_np**2 - np.exp(z_log_np), axis=1)
        scores.append(recon_err + beta * kl_term)
    return np.concatenate(scores, axis=0)


### ========================= Plotters =========================

def plot_resnet_roc(model, test_ds, num_classes=10):
    
    # 1. Collect ground-truth labels and model probabilities
    y_true = []
    y_prob = []
    for batch_imgs, batch_lbls in test_ds:              # test_ds = pre-processed + batched
        y_true.append(batch_lbls.numpy())
        y_prob.append(model.predict(batch_imgs, verbose=0))

    y_true = np.concatenate(y_true)                    # shape (N,)
    y_prob = np.concatenate(y_prob)                    # shape (N, 10)

    # 2. Binarise labels for one-vs-rest ROC
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # 3. Per-class ROC & AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for c in range(num_classes):
        fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    # 4. Micro- & macro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 5. Plot
    plt.figure(figsize=(12,6))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"micro-avg AUC = {roc_auc['micro']:.3f}", lw=2)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f"macro-avg AUC = {roc_auc['macro']:.3f}", lw=2, linestyle="--")

    for c in range(num_classes):
        plt.plot(fpr[c], tpr[c], alpha=0.3,
                 label=f"class {c} AUC = {roc_auc[c]:.2f}")

    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC curves")
    plt.legend(bbox_to_anchor=(1.05,1.0), loc="upper left")
    plt.tight_layout(); plt.show()



def plot_dnn_roc(model, test_dataset, num_classes=10, figsize=(10, 6)):
    """
    Plots ROC curves for a multi-class classifier in a one-vs-rest fashion,
    including a micro-average curve.
    
    Parameters:
    - model: Trained Keras model.
    - test_dataset: tf.data.Dataset yielding (features, labels).
    - num_classes: Number of classes.
    """
    # 1. Collect all true labels and predicted probabilities
    y_true = []
    y_pred = []
    for X_batch, y_batch in test_dataset:
        preds = model.predict(X_batch)
        y_true.append(y_batch.numpy())
        y_pred.append(preds)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    # 2. Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    # 3. Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 4. Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 5. Plot all ROC curves
    plt.figure(figsize=figsize)
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"micro-average (AUC = {roc_auc['micro']:.2f})")
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i],
                 label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    
    # Diagonal line for random guessing
    plt.plot([0, 1], [0, 1], linestyle="--")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curves")
    plt.legend(loc="best")
    plt.show()


# this was used for the DNN and also for the ResNet50 models
def plot_confusion_matrix(
    model,
    test_pca_ds,
    class_names=None,
    normalize="true"
):
    """
    Predict on `test_pca_ds` (features are 512-dim PCA vectors),
    build a confusion matrix and plot it.

    Args
    ----
    model        : trained tf.keras.Model.
    test_pca_ds  : tf.data.Dataset yielding (features, labels),
                   where features.shape = (batch_size, 512)
    class_names  : list of str, length = #classes; defaults to "0", "1", …
    normalize    : {"true","pred","all", None} → same options as sklearn.
    """
    # 1) Pull all true labels into one array
    y_true = np.concatenate([
        labels.numpy()
        for features, labels in test_pca_ds
    ])

    # 2) Predict on all batches of features
    #    We map away the labels so predict() only sees (batch,512) arrays
    feature_ds = test_pca_ds.map(lambda features, labels: features)
    preds = model.predict(feature_ds, verbose=0)
    y_pred = np.argmax(preds, axis=-1)

    # 3) Build the confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # 4) Default class names if none provided
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # 5) Plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    title = "Confusion matrix"
    if normalize:
        title += f" (normalized = '{normalize}')"
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(feats_norm, feats_ano, n_components=2):
    """
    2D scatter of PCA projection.
    """
    all_feats = np.vstack([feats_norm, feats_ano])
    labels   = np.concatenate([np.zeros(len(feats_norm)), np.ones(len(feats_ano))])
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(all_feats)

    plt.figure(figsize=(6,5))
    plt.scatter(proj[labels==0,0], proj[labels==0,1], s=10, alpha=0.6, label='Normal')
    plt.scatter(proj[labels==1,0], proj[labels==1,1], s=10, alpha=0.6, label='Anomaly')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA ({n_components}D → 2D)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_umap_scatter(feats_norm, feats_ano, n_neighbors=15, min_dist=0.1):
    """
    2D scatter of UMAP projection.
    """
    all_feats = np.vstack([feats_norm, feats_ano])
    labels   = np.concatenate([np.zeros(len(feats_norm)), np.ones(len(feats_ano))])
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    proj = reducer.fit_transform(all_feats)

    plt.figure(figsize=(6,5))
    plt.scatter(proj[labels==0,0], proj[labels==0,1], s=10, alpha=0.6, label='Normal')
    plt.scatter(proj[labels==1,0], proj[labels==1,1], s=10, alpha=0.6, label='Anomaly')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_umap_classifier_comparison(feats1, labels1, feats2, labels2, class_names=None,
                         n_neighbors=15, min_dist=0.1,
                         title1='Model 1', title2='Model 2'):
    """
    Side-by-side UMAP comparison of two feature + label sets.

    Parameters:
    - feats1, feats2: numpy arrays of shape (N, D)
    - labels1, labels2: arrays of length N with integer class labels
    - class_names: list of string names for each class (optional)
    - n_neighbors, min_dist: UMAP hyperparameters
    - title1, title2: titles for the left/right plots
    """
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=0)

    # Fit UMAP on concatenation to ensure comparable embeddings
    all_feats = np.vstack([feats1, feats2])
    proj_all = reducer.fit_transform(all_feats)
    n1 = len(feats1)
    proj1 = proj_all[:n1]
    proj2 = proj_all[n1:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, proj, labels, title in zip(axes,
                                       [proj1, proj2],
                                       [labels1, labels2],
                                       [title1, title2]):
        scatter = ax.scatter(proj[:,0], proj[:,1], c=labels,
                             cmap='tab10', s=15, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        if class_names is not None:
            handles = scatter.legend_elements()[0]
            ax.legend(handles, class_names, title="Classes",
                      bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle('UMAP Comparison')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def plot_tsne_scatter(feats_norm, feats_ano, perplexity=30, n_iter=1000):
    """
    2D scatter of t-SNE projection.
    """
    all_feats = np.vstack([feats_norm, feats_ano])
    labels   = np.concatenate([np.zeros(len(feats_norm)), np.ones(len(feats_ano))])
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    proj = tsne.fit_transform(all_feats)

    plt.figure(figsize=(6,5))
    plt.scatter(proj[labels==0,0], proj[labels==0,1], s=10, alpha=0.6, label='Normal')
    plt.scatter(proj[labels==1,0], proj[labels==1,1], s=10, alpha=0.6, label='Anomaly')
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.title(f't-SNE (perplexity={perplexity}, n_iter={n_iter})')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_anom_umap_per_model(models_scores_norm, models_scores_ano, model_names,
                        n_neighbors=15, min_dist=0.1):
    """
    For each model, perform UMAP on its anomaly scores and plot separately
    in a 2x2 grid.

    Parameters:
    - models_scores_norm: list of numpy arrays; each array is scores_norm for one model
    - models_scores_ano:  list of numpy arrays; each array is scores_ano for one model
    - model_names:         list of strings, model names (same order)
    - n_neighbors, min_dist: UMAP parameters
    """
    num_models = len(model_names)
    # Create 2 rows x 2 cols grid
    fig, axes = plt.subplots(2, num_models // 2, figsize=(12, 8), squeeze=False)
    axes = axes.flatten()

    ## colormap
    orig = plt.get_cmap('berlin')(np.linspace(0, 1, 256))  # shape (256, 4)

    # 2) Scale the RGB channels (cols 0–2) by a factor < 1 to darken
    factor = 0.8  # 0 = black; 1 = original brightness
    darker = orig.copy()
    darker[:, :3] = darker[:, :3] * factor
    darker = np.clip(darker, 0, 1)  # ensure valid range

    # 3) Create a new ListedColormap
    dark_managua = ListedColormap(darker, name='dark_berlin')

    for ax, scores_norm, scores_ano, name in zip(axes, models_scores_norm, models_scores_ano, model_names):
        # Combine normal and anomaly scores into a (N,1) feature
        X = np.concatenate([scores_norm, scores_ano]).reshape(-1, 1)
        y = np.concatenate([np.zeros(len(scores_norm)), np.ones(len(scores_ano))])

        # UMAP embedding on 1-D scores
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=min_dist)
        embedding = reducer.fit_transform(X)

        # Scatter plot colored by true label
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=dark_managua,
                             s=15, alpha=0.5)
        ax.set_title(f"{name} UMAP")
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        # Legend for normal/anomaly
        handles = scatter.legend_elements()[0]
        ax.legend(handles, ['Normal', 'Anomaly'], title="True Label", loc='best')

    plt.tight_layout()
    plt.show()

## ___________ maf evaluation ___________
def evaluate_to_visual(
    maf_model, 
    feats_norm, feats_ano
):
    """
    Compute AUROC and optionally plot.
    """
    # scores
    scores_norm = scores_from_embeddings(maf_model, feats_norm)
    scores_ano  = scores_from_embeddings(maf_model, feats_ano)

    # AUROC
    auc_score =  roc_auc_score(
        np.concatenate([np.zeros_like(scores_norm), np.ones_like(scores_ano)]),
        np.concatenate([scores_norm, scores_ano])
    )
    print(f"AUROC = {auc_score:.4f}")

    return scores_norm, scores_ano, auc_score


def plot_score_histograms(scores_norm, scores_ano, norm_bins=40, anom_bins=40):
    """
    Plot log-scale histograms of normal vs anomaly scores.
    """
    plt.figure(figsize=(10,6))
    plt.hist(scores_norm, bins=norm_bins, alpha=0.6, label='Normal')
    plt.hist(scores_ano,  bins=anom_bins, alpha=0.6, label='Anomaly')
    plt.yscale('log')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count (log scale)')
    plt.title('Score Distributions')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_maf_roc_curve(scores_norm, scores_ano):
    """
    Plot ROC of normal vs anomaly scores.
    """
    y_true = np.concatenate([np.zeros_like(scores_norm), np.ones_like(scores_ano)])
    y_scores = np.concatenate([scores_norm, scores_ano])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    return roc_auc

def scores_from_embeddings(maf_model, embeddings):
    """
    Given an (N, D) array of embeddings, returns anomaly scores = -log_prob.
    """
    logp = maf_model.predict(embeddings, verbose=0).flatten()
    return -logp



def plot_marginal_differences(model, ds_train):
    # 1) sample synthetic points
    num_synth = 5000
    x_synth = model.flow.sample(num_synth).numpy()  # (5000, 32)

    # 2) load *all* real embeddings once (no limit)
    real_embs = np.stack(list(ds_train.unbatch().as_numpy_iterator()))
    N_real   = real_embs.shape[0]

    # 3) normalize both together
    scaler     = StandardScaler().fit(np.vstack([real_embs, x_synth]))
    real_norm  = scaler.transform(real_embs)
    synth_norm = scaler.transform(x_synth)

    # 4) compute *density* histograms
    bins = np.linspace(-4, 4, 50)
    mids = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(4, 8, figsize=(24, 16), sharex=True, sharey=True)
    fig.suptitle("Real-Synthetic PDFs (normalized)", fontsize=16)
    for dim, ax in enumerate(axes.flat):
        real_pdf, _  = np.histogram(real_norm[:, dim],  bins=bins, density=True)
        synth_pdf, _ = np.histogram(synth_norm[:, dim], bins=bins, density=True)
        diff_pdf = real_pdf - synth_pdf

        ax.bar(mids, diff_pdf, width=(bins[1]-bins[0]), edgecolor='k', alpha=0.7)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(f"Dim {dim}", fontsize=10)
        if dim % 8 == 0:
            ax.set_ylabel("PDF(real)-PDF(flow)")
        if dim >= 24:
            ax.set_xlabel("Normalized value")

    plt.tight_layout()
    plt.show()
