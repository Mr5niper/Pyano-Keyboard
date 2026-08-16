"""
Extract the largest PNG-compressed image from icon.ico and write it as
icon_win.png.

Why: pygame/SDL cannot decode PNG-compressed .ico entries (which most modern
icon editors export), so the app loads its window icon from a plain PNG instead.
BUILD_EXE.bat runs this before building so icon_win.png always matches the
current icon.ico. You never have to maintain the PNG by hand; just update
icon.ico as usual.

If icon.ico contains no PNG-compressed entry (older BMP-style .ico), any existing
icon_win.png is left untouched.
"""

import struct
import sys


def main():
    ico = sys.argv[1] if len(sys.argv) > 1 else "icon.ico"
    out = sys.argv[2] if len(sys.argv) > 2 else "icon_win.png"
    try:
        data = open(ico, "rb").read()
    except Exception as e:
        print(f"  could not read {ico}: {e}")
        return 0

    if len(data) < 6:
        print(f"  {ico} is not a valid icon file")
        return 0

    reserved, itype, count = struct.unpack_from("<HHH", data, 0)
    off = 6
    best = None  # (width, png_bytes)
    for _ in range(count):
        if off + 16 > len(data):
            break
        w, h, ncol, rsv, planes, bpp, size, offset = struct.unpack_from(
            "<BBBBHHII", data, off)
        w = w or 256
        entry = data[offset:offset + size]
        if entry[:8] == b"\x89PNG\r\n\x1a\n" and (best is None or w > best[0]):
            best = (w, entry)
        off += 16

    if best:
        try:
            with open(out, "wb") as f:
                f.write(best[1])
            print(f"  wrote {out} ({best[0]}x{best[0]})")
        except Exception as e:
            print(f"  could not write {out}: {e}")
    else:
        print(f"  no PNG entry in {ico}; keeping existing {out} if any")
    return 0


if __name__ == "__main__":
    sys.exit(main())
