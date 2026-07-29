"""Universal import & converter.

Accepts almost anything, figures out what it is, and turns it into CSV (or routes
an image dataset to the vision engine):

  • tabular file  (csv / xlsx / xls / json / tsv / txt / parquet)  → CSV
  • data zip      (a zip containing tabular files)                 → pick one → CSV
  • image zip     (folders of images)                              → metadata CSV
                                                                     or train a CNN
"""

import io
import zipfile

import pandas as pd

DATA_EXTS = {"csv", "xlsx", "xls", "json", "tsv", "txt", "parquet"}
IMG_EXTS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}

# Holds the most-recent upload + last converted CSV (single-user, like data_store)
CONVERT_STORE = {}


def _ext(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _clean_parts(name: str):
    return [p for p in name.replace("\\", "/").split("/") if p and not p.startswith("__MACOSX")]


def _read_tabular(content: bytes, ext: str, sheet=None) -> pd.DataFrame:
    bio = io.BytesIO(content)
    if ext == "csv":
        return pd.read_csv(bio)
    if ext in ("xlsx", "xls"):
        return pd.read_excel(bio, sheet_name=sheet if sheet is not None else 0)
    if ext == "json":
        return pd.read_json(bio)
    if ext == "tsv":
        return pd.read_csv(bio, sep="\t")
    if ext == "txt":
        # sniff the delimiter
        return pd.read_csv(bio, sep=None, engine="python")
    if ext == "parquet":
        try:
            return pd.read_parquet(bio)
        except Exception as e:
            raise ValueError(f"Reading parquet needs the 'pyarrow' package: {e}")
    raise ValueError(f"Unsupported tabular format: .{ext}")


def _preview(df: pd.DataFrame) -> dict:
    return {
        "columns": [str(c) for c in df.columns],
        "rows": df.head(10).fillna("").astype(str).to_dict(orient="records"),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
    }


def _excel_sheets(content: bytes):
    try:
        xls = pd.ExcelFile(io.BytesIO(content))
        return xls.sheet_names
    except Exception:
        return None


def inspect_upload(content: bytes, filename: str) -> dict:
    """Detect what was uploaded and describe the actions available."""
    CONVERT_STORE.clear()
    CONVERT_STORE.update({"content": content, "filename": filename})
    ext = _ext(filename)

    if ext == "zip":
        return _inspect_zip(content, filename)

    if ext in DATA_EXTS:
        try:
            sheets = _excel_sheets(content) if ext in ("xlsx", "xls") else None
            df = _read_tabular(content, ext, sheet=(sheets[0] if sheets else None))
        except Exception as e:
            return {"kind": "unsupported", "filename": filename, "note": str(e)}
        return {
            "kind": "tabular", "filename": filename, "format": ext,
            "sheets": sheets, "preview": _preview(df),
            "already_csv": ext == "csv",
        }

    if ext in IMG_EXTS:
        return {"kind": "image_single", "filename": filename,
                "note": "That's a single image. Zip a set of labelled image folders to train a classifier."}

    return {"kind": "unsupported", "filename": filename,
            "note": f"Don't know how to convert a .{ext or '?'} file."}


def _inspect_zip(content: bytes, filename: str) -> dict:
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return {"kind": "unsupported", "filename": filename, "note": "That isn't a valid .zip archive."}

    data_files, img_by_class = [], {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = _clean_parts(info.filename)
        if not parts:
            continue
        ext = _ext(parts[-1])
        if ext in DATA_EXTS:
            data_files.append({"name": info.filename, "format": ext, "bytes": info.file_size})
        elif ext in IMG_EXTS and len(parts) >= 2:
            img_by_class.setdefault(parts[-2], 0)
            img_by_class[parts[-2]] += 1

    n_imgs = sum(img_by_class.values())
    if n_imgs >= 2 and n_imgs >= len(data_files):
        trainable_classes = {c: n for c, n in img_by_class.items() if n >= 2}
        return {
            "kind": "image_zip", "filename": filename,
            "classes": dict(sorted(img_by_class.items())),
            "total": n_imgs,
            "trainable": len(trainable_classes) >= 2,
        }

    if data_files:
        return {"kind": "data_zip", "filename": filename, "files": data_files}

    return {"kind": "unsupported", "filename": filename,
            "note": "The zip has no recognizable data files or image folders."}


def convert(choice: dict | None = None) -> dict:
    """Convert the stored upload to CSV. `choice` selects a zip member or a
    workbook sheet. Stores the CSV bytes for download and returns a preview."""
    if "content" not in CONVERT_STORE:
        return {"status": "error", "note": "Nothing uploaded to convert."}
    content, filename = CONVERT_STORE["content"], CONVERT_STORE["filename"]
    ext = _ext(filename)
    choice = choice or {}

    try:
        if ext == "zip":
            member = choice.get("file")
            if not member:
                return {"status": "error", "note": "Choose a file inside the zip to convert."}
            zf = zipfile.ZipFile(io.BytesIO(content))
            raw = zf.read(member)
            df = _read_tabular(raw, _ext(member))
            out_name = _clean_parts(member)[-1].rsplit(".", 1)[0]
        else:
            df = _read_tabular(content, ext, sheet=choice.get("sheet"))
            out_name = filename.rsplit(".", 1)[0]
    except Exception as e:
        return {"status": "error", "note": str(e)}

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    CONVERT_STORE["csv"] = csv_bytes
    CONVERT_STORE["csv_name"] = f"{out_name}.csv"
    CONVERT_STORE["df"] = df
    return {"status": "success", "csv_name": CONVERT_STORE["csv_name"], "preview": _preview(df)}


def image_metadata(content: bytes | None = None) -> dict:
    """Build a CSV describing every image in the stored (or given) zip."""
    from PIL import Image
    content = content or CONVERT_STORE.get("content")
    if not content:
        return {"status": "error", "note": "No image archive uploaded."}
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return {"status": "error", "note": "That isn't a valid .zip archive."}

    rows = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = _clean_parts(info.filename)
        if not parts or _ext(parts[-1]) not in IMG_EXTS:
            continue
        cls = parts[-2] if len(parts) >= 2 else ""
        row = {"filename": parts[-1], "class": cls, "bytes": info.file_size,
               "width": None, "height": None, "mode": None}
        try:
            with zf.open(info) as fh:
                im = Image.open(io.BytesIO(fh.read()))
                row["width"], row["height"], row["mode"] = im.width, im.height, im.mode
        except Exception:
            pass
        rows.append(row)
        if len(rows) >= 5000:
            break

    if not rows:
        return {"status": "error", "note": "No images found in the archive."}
    df = pd.DataFrame(rows, columns=["filename", "class", "width", "height", "mode", "bytes"])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    name = CONVERT_STORE.get("filename", "images").rsplit(".", 1)[0]
    CONVERT_STORE["csv"] = csv_bytes
    CONVERT_STORE["csv_name"] = f"{name}_metadata.csv"
    CONVERT_STORE["df"] = df
    return {"status": "success", "csv_name": CONVERT_STORE["csv_name"], "preview": _preview(df)}


def get_converted():
    """Return (csv_bytes, name, df) for the last conversion, or (None, None, None)."""
    return CONVERT_STORE.get("csv"), CONVERT_STORE.get("csv_name"), CONVERT_STORE.get("df")


def get_stored_zip():
    """Raw bytes of the last uploaded archive (for routing to the vision engine)."""
    return CONVERT_STORE.get("content")
