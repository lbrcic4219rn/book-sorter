"""Minimal MOBI/AZW/AZW3/PRC reader: first few text records only (PalmDOC compression).
Much faster than `ebook-convert`; returns "" for anything it can't handle (e.g. HUFF/CDIC,
DRM), so the caller can fall back to Calibre."""
import struct
from pathlib import Path


def _palmdoc_decompress(data: bytes) -> bytes:
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        i += 1
        if c == 0 or 0x09 <= c <= 0x7F:
            out.append(c)
        elif 0x01 <= c <= 0x08:
            out += data[i:i + c]
            i += c
        elif 0x80 <= c <= 0xBF:
            if i >= n:
                break
            pair = (c << 8) | data[i]
            i += 1
            dist, length = (pair >> 3) & 0x07FF, (pair & 0x07) + 3
            if dist == 0 or dist > len(out):
                break
            for _ in range(length):
                out.append(out[-dist])
        else:  # 0xC0-0xFF: space + char
            out += b" " + bytes([c ^ 0x80])
    return bytes(out)


def _trailing_size(rec: bytes, flags: int) -> int:
    """Size of the trailing entries MOBI appends to each text record."""
    size = 0
    for bit in range(15, 0, -1):
        if flags & (1 << bit):
            # backward-encoded varint at the end of the (remaining) record
            val, shift = 0, 0
            for b in reversed(rec[: len(rec) - size][-4:]):
                val |= (b & 0x7F) << shift
                shift += 7
                if b & 0x80:
                    break
            size += val
    if flags & 1:
        size += (rec[len(rec) - size - 1] & 0x3) + 1
    return size


def mobi_text(path: Path, max_records: int = 6) -> str:
    try:
        raw = path.read_bytes()
        n_recs = struct.unpack(">H", raw[76:78])[0]
        offsets = [struct.unpack(">I", raw[78 + 8 * i: 82 + 8 * i])[0] for i in range(n_recs)] + [len(raw)]
        rec0 = raw[offsets[0]:offsets[1]]
        compression, _, _, text_recs = struct.unpack(">HHIH", rec0[:10])
        if compression not in (1, 2) or rec0[16:20] != b"MOBI":
            return ""
        if struct.unpack(">H", rec0[12:14])[0]:  # encryption
            return ""
        mobi_len = struct.unpack(">I", rec0[20:24])[0]
        encoding = struct.unpack(">I", rec0[28:32])[0]
        flags = struct.unpack(">H", rec0[0xF2:0xF4])[0] if mobi_len >= 0xE4 else 0
        chunks = []
        for i in range(1, min(text_recs, max_records) + 1):
            rec = raw[offsets[i]:offsets[i + 1]]
            rec = rec[: len(rec) - _trailing_size(rec, flags)]
            chunks.append(_palmdoc_decompress(rec) if compression == 2 else rec)
        html = b"".join(chunks).decode("utf-8" if encoding == 65001 else "cp1252", "ignore")
        return html
    except (struct.error, IndexError, OSError):
        return ""
