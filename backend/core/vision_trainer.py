"""CNN image-classification engine.

Accepts a .zip whose sub-folders are class labels (cats/…, dogs/…), trains an
image classifier, and serves live single-image predictions.

Strategy — transfer learning:
  A frozen, pretrained MobileNetV2 backbone turns each image into a 1280-dim
  embedding once; a small linear head is then trained on those embeddings with a
  live per-epoch loss/accuracy curve. This is fast even on CPU and far more
  accurate on small datasets than training from scratch. If the pretrained
  weights can't be downloaded (offline), we fall back to a small CNN trained
  end-to-end so the feature still works.

All heavy imports are guarded so the module loads even without torch installed.
"""

import io
import time
import base64
import zipfile

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import transforms, models
    HAS_TORCH = True
except Exception:  # torch or torchvision missing
    HAS_TORCH = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Images work with Pillow alone (lightweight, deploy-safe); torch adds the CNN.
VISION_AVAILABLE = HAS_PIL
ENGINE_NAME = "PyTorch (CNN)" if HAS_TORCH else "scikit-learn (image features)"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
IMG_SIZE = 128
MAX_PER_CLASS = 200
MAX_TOTAL = 1200
THUMBS_PER_CLASS = 4

# In-memory dataset + trained model (single-user app, mirrors data_store)
VISION_STORE = {}   # {classes, counts, images:[(PIL,label_idx)], total, thumbnails, warnings}
VISION_MODEL = {}   # {backbone, head, transform, classes, mode}

_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


def _thumb_data_uri(img, size=96) -> str:
    im = img.copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _clean_parts(name: str):
    parts = [p for p in name.replace("\\", "/").split("/") if p and not p.startswith("__MACOSX")]
    return parts


def load_image_zip(content: bytes) -> dict:
    """Unpack a zip of images. The immediate parent folder of each image is its
    class label — robust to an extra wrapper folder (dataset/cats/1.jpg)."""
    if not HAS_PIL:
        return {"status": "error", "note": "Image tools need Pillow installed on the backend."}

    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return {"status": "error", "note": "That file isn't a valid .zip archive."}

    by_class = {}
    warnings = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = _clean_parts(info.filename)
        if len(parts) < 2:
            continue  # image sitting at the zip root has no class folder
        fname = parts[-1]
        ext = "." + fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext not in IMG_EXTS:
            continue
        label = parts[-2]
        by_class.setdefault(label, [])
        if len(by_class[label]) >= MAX_PER_CLASS:
            continue
        try:
            with zf.open(info) as fh:
                img = Image.open(io.BytesIO(fh.read())).convert("RGB")
            img.thumbnail((256, 256))
            by_class[label].append(img)
        except Exception:
            warnings.append(f"Skipped unreadable image: {fname}")

    # keep only classes with enough images
    by_class = {c: imgs for c, imgs in by_class.items() if len(imgs) >= 2}
    if len(by_class) < 2:
        return {"status": "error",
                "note": "Need at least 2 class folders (each with ≥ 2 images). "
                        "Structure the zip as one folder per class, e.g. cats/…, dogs/…"}

    # cap total, build flat sample list
    classes = sorted(by_class.keys())
    images, counts, thumbnails = [], {}, {}
    total_budget = MAX_TOTAL
    for ci, cls in enumerate(classes):
        imgs = by_class[cls][: max(2, total_budget // (len(classes) - ci))]
        counts[cls] = len(imgs)
        total_budget -= len(imgs)
        thumbnails[cls] = [_thumb_data_uri(im) for im in imgs[:THUMBS_PER_CLASS]]
        for im in imgs:
            images.append((im, ci))

    VISION_STORE.clear()
    VISION_STORE.update({
        "classes": classes, "counts": counts, "images": images,
        "total": len(images), "thumbnails": thumbnails, "warnings": warnings[:10],
    })
    VISION_MODEL.clear()  # invalidate any previous model

    return {
        "status": "success", "classes": classes, "counts": counts,
        "total": len(images), "thumbnails": thumbnails,
        "warnings": warnings[:10], "engine": ENGINE_NAME,
    }


def current_dataset():
    """Summary of the dataset currently loaded in memory, or None."""
    if not VISION_STORE.get("images"):
        return None
    return {
        "status": "success",
        "classes": VISION_STORE["classes"],
        "counts": VISION_STORE["counts"],
        "total": VISION_STORE["total"],
        "thumbnails": VISION_STORE["thumbnails"],
        "warnings": VISION_STORE.get("warnings", []),
        "engine": ENGINE_NAME,
    }


def _build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])


def _try_backbone():
    """Return (feature_fn, out_dim, mode). Uses pretrained MobileNetV2 when the
    weights download succeeds, else signals the from-scratch fallback."""
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        net = models.mobilenet_v2(weights=weights)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        pool = nn.AdaptiveAvgPool2d(1)

        def feat(x):
            with torch.no_grad():
                return pool(net.features(x)).flatten(1)
        return feat, 1280, net, "transfer"
    except Exception:
        return None, 0, None, "scratch"


if HAS_TORCH:
    class _SmallCNN(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(64, n_classes)

        def forward(self, x):
            return self.head(self.net(x).flatten(1))


def train_classifier(config: dict | None = None) -> dict:
    if not HAS_PIL:
        return {"status": "error", "note": "Image tools need Pillow installed on the backend."}
    if not VISION_STORE.get("images"):
        return {"status": "error", "note": "Upload a zip of labelled images first."}
    return _train_cnn(config) if HAS_TORCH else _train_lite(config)


def _train_cnn(config: dict | None = None) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = {"epochs": 15, "learning_rate": 0.001, **(config or {})}
    epochs = int(np.clip(cfg["epochs"], 1, 60))
    lr = float(np.clip(cfg["learning_rate"], 1e-4, 0.1))

    classes = VISION_STORE["classes"]
    images = VISION_STORE["images"]
    transform = _build_transform()
    X = torch.stack([transform(im) for im, _ in images])          # (N,3,H,W)
    y = torch.tensor([lbl for _, lbl in images], dtype=torch.long)

    feat_fn, out_dim, backbone, mode = _try_backbone()
    t0 = time.time()

    if mode == "transfer":
        # embed all images once through the frozen backbone
        embeds = []
        for i in range(0, len(X), 32):
            embeds.append(feat_fn(X[i:i + 32]))
        Z = torch.cat(embeds)
        model = nn.Linear(out_dim, len(classes))
        train_input = Z
    else:  # from-scratch small CNN trained end-to-end on the images
        model = _SmallCNN(len(classes))
        train_input = X

    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=max(0.2, min(0.3, 30 / len(idx))),
                              random_state=42, stratify=y.numpy())
    Xtr, Xva = train_input[tr], train_input[va]
    ytr, yva = y[tr], y[va]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    batch = 32
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        ep_loss, nb = 0.0, 0
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            optimizer.zero_grad()
            loss = criterion(model(Xtr[b]), ytr[b])
            loss.backward(); optimizer.step()
            ep_loss += float(loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = float(criterion(val_logits, yva).item())
            val_pred = val_logits.argmax(1).numpy()
            val_acc = float(accuracy_score(yva.numpy(), val_pred))
        history.append({"epoch": epoch + 1, "train_loss": round(ep_loss / max(1, nb), 4),
                        "val_loss": round(val_loss, 4), "val_metric": round(val_acc, 4)})

    elapsed = round(time.time() - t0, 2)

    model.eval()
    with torch.no_grad():
        logits = model(Xva)
        y_pred = logits.argmax(1).numpy()
    y_true = yva.numpy()

    n_classes = len(classes)
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()

    n_params = sum(p.numel() for p in model.parameters())
    VISION_MODEL.clear()
    VISION_MODEL.update({
        "model": model, "feat_fn": feat_fn, "transform": transform,
        "classes": classes, "mode": mode,
    })

    return {
        "status": "success", "engine": "PyTorch",
        "backbone": "MobileNetV2 (pretrained, frozen)" if mode == "transfer" else "Small CNN (from scratch)",
        "mode": mode, "classes": classes, "metrics": metrics,
        "primary_metric": "accuracy", "metric_label": "Accuracy",
        "history": history, "confusion_matrix": cm, "labels": classes,
        "n_params": int(n_params), "training_time": f"{elapsed}s",
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "architecture": (["Input · 128×128×3 image",
                          "MobileNetV2 features · frozen",
                          "Global average pool · 1280-d embedding",
                          f"Dense · {n_classes} classes (softmax)"] if mode == "transfer"
                         else ["Input · 128×128×3 image",
                               "Conv 16 · BN · ReLU · MaxPool",
                               "Conv 32 · BN · ReLU · MaxPool",
                               "Conv 64 · BN · ReLU · GAP",
                               f"Dense · {n_classes} classes (softmax)"]),
    }


# ── Lightweight engine (no torch): image features + scikit-learn ──────────────
def _img_features(img) -> np.ndarray:
    """A compact colour/texture descriptor for one image (works with Pillow only)."""
    im = img.convert("RGB").resize((32, 32))
    arr = np.asarray(im, dtype=float) / 255.0            # (32,32,3)
    feats = []
    for c in range(3):                                    # per-channel mean & std
        ch = arr[:, :, c]
        feats += [ch.mean(), ch.std()]
    for c in range(3):                                    # per-channel 8-bin histogram
        h, _ = np.histogram(arr[:, :, c], bins=8, range=(0, 1))
        s = h.sum()
        feats += (h / s).tolist() if s else h.tolist()
    gray = arr.mean(axis=2)                               # simple texture: gradient energy
    feats += [float(np.abs(np.diff(gray, axis=1)).mean()),
              float(np.abs(np.diff(gray, axis=0)).mean())]
    feats += [img.width / max(1, img.height), float(gray.mean())]  # aspect ratio, brightness
    return np.array(feats, dtype=float)


def _train_lite(config: dict | None = None) -> dict:
    np.random.seed(42)
    cfg = {"epochs": 15, **(config or {})}
    epochs = int(np.clip(cfg["epochs"], 1, 60))

    classes = VISION_STORE["classes"]
    images = VISION_STORE["images"]
    X = np.array([_img_features(im) for im, _ in images])
    y = np.array([lbl for _, lbl in images])

    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=max(0.2, min(0.3, 30 / len(idx))),
                              random_state=42, stratify=y)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    ytr, yva = y[tr], y[va]

    model = MLPClassifier(hidden_layer_sizes=(64,), learning_rate_init=0.01, random_state=42)
    all_classes = np.unique(y)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.partial_fit(Xtr, ytr, classes=all_classes)
        val_acc = float(accuracy_score(yva, model.predict(Xva)))
        history.append({"epoch": epoch + 1, "train_loss": round(float(model.loss_), 4),
                        "val_loss": None, "val_metric": round(val_acc, 4)})
    elapsed = round(time.time() - t0, 2)

    y_pred = model.predict(Xva)
    n_classes = len(classes)
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "accuracy": round(float(accuracy_score(yva, y_pred)), 4),
        "f1_score": round(float(f1_score(yva, y_pred, average=avg, zero_division=0)), 4),
    }
    cm = confusion_matrix(yva, y_pred, labels=list(range(n_classes))).tolist()
    n_params = int(sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_))

    VISION_MODEL.clear()
    VISION_MODEL.update({"model": model, "scaler": scaler, "classes": classes, "mode": "lite"})
    return {
        "status": "success", "engine": ENGINE_NAME,
        "backbone": "Colour & texture features + scikit-learn MLP",
        "mode": "lite", "classes": classes, "metrics": metrics,
        "primary_metric": "accuracy", "metric_label": "Accuracy",
        "history": history, "confusion_matrix": cm, "labels": classes,
        "n_params": n_params, "training_time": f"{elapsed}s",
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "architecture": [f"Input · image → {X.shape[1]} colour/texture features",
                         "Standardize", "Dense 64 · ReLU",
                         f"Dense · {n_classes} classes (softmax)"],
    }


def predict_image(content: bytes) -> dict:
    if not HAS_PIL:
        return {"status": "error", "note": "Image tools need Pillow installed on the backend."}
    if not VISION_MODEL.get("model"):
        return {"status": "error", "note": "Train an image classifier first."}
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return {"status": "error", "note": "Could not read that image file."}

    if VISION_MODEL["mode"] == "lite":
        feats = VISION_MODEL["scaler"].transform(_img_features(img).reshape(1, -1))
        proba = VISION_MODEL["model"].predict_proba(feats)[0]
    else:
        x = VISION_MODEL["transform"](img).unsqueeze(0)
        model = VISION_MODEL["model"]
        model.eval()
        with torch.no_grad():
            fx = VISION_MODEL["feat_fn"](x) if VISION_MODEL["mode"] == "transfer" else x
            proba = torch.softmax(model(fx), dim=1)[0].numpy()

    classes = VISION_MODEL["classes"]
    order = np.argsort(proba)[::-1]
    dist = [{"class": classes[i], "prob": round(float(proba[i]), 4)} for i in order]
    return {
        "status": "success",
        "prediction": classes[int(order[0])],
        "confidence": round(float(proba[order[0]]), 4),
        "distribution": dist,
        "thumbnail": _thumb_data_uri(img, size=160),
    }
