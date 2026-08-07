"""Bounded reading of uploaded files.

Starlette has already written the whole request body to a `SpooledTemporaryFile`
by the time a handler runs — in memory below 1 MB, on disk above it. That is the
important fact this module is built around: for anything large, the upload is
*already on disk*, and calling `await file.read()` does not fetch it so much as
copy it into RAM a second time.

That copy is what took the container down. A 290 MB image zip read this way cost
290 MB of resident memory before a single image had been decoded, and on a 512 MB
box the decode that followed pushed it past the limit. The process was killed
mid-request, which the browser saw as a 502.

So: check the size without reading, then hand the file object itself to whatever
consumes it. `zipfile` and `pandas` both accept a file object and seek around it
happily, and neither needs the bytes in memory to do so.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, UploadFile

# Tabular uploads. These are parsed into a DataFrame that lives in memory for the
# session, so the ceiling is about what the box can hold, not just transfer size.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# Archives. Allowed to be far larger because they are never held in memory whole:
# the zip is read from disk and only the images that survive the per-class cap are
# ever decoded.
MAX_ZIP_MB = int(os.getenv("MAX_ZIP_MB", "300"))

_CHUNK = 1024 * 1024


def upload_size(upload: UploadFile) -> int:
    """Byte length of an upload, without reading it into memory."""
    f = upload.file
    pos = f.tell()
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(pos)
    return size


def enforce_size(upload: UploadFile, max_mb: int, what: str = "file") -> int:
    """Reject an oversized upload before anything touches its contents."""
    size = upload_size(upload)
    if size == 0:
        raise HTTPException(status_code=400, detail=f"That {what} is empty.")
    if size > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"That {what} is {size / 1024 / 1024:.0f} MB, over the {max_mb} MB "
                   f"limit. Upload a smaller archive, or fewer images per class.",
        )
    return size


def rewound(upload: UploadFile) -> "object":
    """The upload's underlying file object, seeked to the start.

    Hand this to zipfile/pandas instead of `await upload.read()`. It is a real
    seekable file, so they can work through it without the bytes ever being
    resident.
    """
    upload.file.seek(0)
    return upload.file


def read_capped(upload: UploadFile, max_mb: int = MAX_UPLOAD_MB, what: str = "file") -> bytes:
    """Read an upload into memory, refusing anything over the cap.

    For consumers that genuinely need bytes. Prefer `rewound()` where the
    consumer can take a file object.
    """
    enforce_size(upload, max_mb, what)
    upload.file.seek(0)
    chunks, total = [], 0
    max_bytes = max_mb * 1024 * 1024
    while True:
        chunk = upload.file.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            # Belt and braces: enforce_size already checked, but a streaming
            # source could still overrun and this bounds it either way.
            raise HTTPException(
                status_code=413,
                detail=f"That {what} is larger than the {max_mb} MB limit.")
        chunks.append(chunk)
    return b"".join(chunks)
