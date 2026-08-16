"""
Raw Input Keyboard Probe  (Windows only)

Why this exists
---------------
The in-app key test reads the keyboard through pygame, which reads it through
SDL, which reads translated Windows keyboard MESSAGES. If a key press is lost
somewhere in that chain, the app never sees it. This probe skips that entire
chain: it registers for the Windows Raw Input API and reads key presses straight
from the HID layer, before SDL or window-message translation touch them.

Run this, then hold the SAME combination that fails in the piano (for example a
+ f + g). Watch the console, and read raw_input_report.txt when you close it.

  - If raw input SHOWS every key you held, but the piano's own test marked some
    as "MISSING", then the loss is in the SDL/message layer and the app can be
    fixed by switching its input to raw input. That is a real, doable change.

  - If raw input ALSO misses the same keys, then the key presses are being lost
    below any software this program can reach (the keyboard controller or the
    USB/HID stack). In that case no code change in the app can recover them.

This uses only the Python standard library (ctypes). No pip installs, no admin.
It opens a tiny hidden window to receive the input; that window does not need to
be focused, but keep this console in the foreground while you test so you know
the keys are going here.

Close the console window (or press Ctrl+C) to stop and write the report.
"""

import ctypes
import ctypes.wintypes as wt
import sys
import time
import datetime

if not sys.platform.startswith("win"):
    print("This probe only runs on Windows.")
    sys.exit(1)

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# --- Win32 constants ---
WM_INPUT = 0x00FF
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
RID_INPUT = 0x10000003
RIM_TYPEKEYBOARD = 1
RIDEV_INPUTSINK = 0x00000100  # receive input even when not focused
RI_KEY_BREAK = 0x01           # key up (else key down)
HWND_MESSAGE = wt.HWND(-3)    # message-only window

# --- Structures ---
class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wt.USHORT),
        ("usUsage", wt.USHORT),
        ("dwFlags", wt.DWORD),
        ("hwndTarget", wt.HWND),
    ]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wt.DWORD),
        ("dwSize", wt.DWORD),
        ("hDevice", wt.HANDLE),
        ("wParam", wt.WPARAM),
    ]

class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wt.USHORT),
        ("Flags", wt.USHORT),
        ("Reserved", wt.USHORT),
        ("VKey", wt.USHORT),
        ("Message", wt.UINT),
        ("ExtraInformation", wt.ULONG),
    ]

class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("keyboard", RAWKEYBOARD),
    ]

WNDPROCTYPE = ctypes.WINFUNCTYPE(
    ctypes.c_long, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM
)

class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", wt.UINT),
        ("lpfnWndProc", WNDPROCTYPE),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wt.HINSTANCE),
        ("hIcon", wt.HANDLE),
        ("hCursor", wt.HANDLE),
        ("hbrBackground", wt.HANDLE),
        ("lpszMenuName", wt.LPCWSTR),
        ("lpszClassName", wt.LPCWSTR),
    ]

# --- Virtual-key code -> readable name (letters/punctuation we care about) ---
VK_NAMES = {}
for c in range(0x30, 0x5B):          # 0-9 and A-Z
    VK_NAMES[c] = chr(c).lower()
VK_NAMES.update({
    0xBA: ";", 0xBB: "=", 0xBC: ",", 0xBD: "-", 0xBE: ".", 0xBF: "/",
    0xC0: "`", 0xDB: "[", 0xDC: "\\", 0xDD: "]", 0xDE: "'",
    0x20: "space", 0x09: "tab", 0x0D: "enter", 0x1B: "esc",
    0x10: "shift", 0x11: "ctrl", 0x12: "alt",
})

def vk_name(vk):
    return VK_NAMES.get(vk, f"vk_{vk}")

events = []          # (t_ms, is_down, vk, makecode, name)
held = {}            # vk -> name, currently physically down
start = time.time()

def now_ms():
    return int((time.time() - start) * 1000)

def handle_rawinput(lparam):
    size = wt.UINT(0)
    user32.GetRawInputData(wt.HANDLE(lparam), RID_INPUT, None,
                           ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
    if size.value == 0:
        return
    buf = ctypes.create_string_buffer(size.value)
    got = user32.GetRawInputData(wt.HANDLE(lparam), RID_INPUT, buf,
                                 ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
    if got != size.value:
        return
    ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
    if ri.header.dwType != RIM_TYPEKEYBOARD:
        return
    kb = ri.keyboard
    vk = kb.VKey
    if vk in (0, 0xFF):   # fake/placeholder keys sent by some keyboards
        return
    is_down = (kb.Flags & RI_KEY_BREAK) == 0
    name = vk_name(vk)
    t = now_ms()
    events.append((t, is_down, vk, kb.MakeCode, name))
    if is_down:
        held[vk] = name
    else:
        held.pop(vk, None)
    held_str = " ".join(sorted(held.values())) if held else "-"
    print(f"{t:>8} ms  {'DOWN' if is_down else 'UP  '}  {name:<8} "
          f"(vk={vk} make={kb.MakeCode})   held now: {held_str}")

def wnd_proc(hwnd, msg, wparam, lparam):
    if msg == WM_INPUT:
        try:
            handle_rawinput(lparam)
        except Exception as e:
            print("error reading raw input:", e)
        return 0
    if msg == WM_DESTROY:
        user32.PostQuitMessage(0)
        return 0
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

def write_report():
    lines = []
    lines.append("Raw Input Keyboard Probe - Report")
    lines.append("Generated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")
    lines.append("This log was captured via the Windows Raw Input API, BEFORE SDL")
    lines.append("or window-message translation. Compare it to the piano's own key")
    lines.append("test report:")
    lines.append("  - Keys that appear HERE but were MISSING in the app  -> the loss")
    lines.append("    is in the SDL/message layer; the app can switch to raw input.")
    lines.append("  - Keys missing HERE too -> lost below all app-reachable software")
    lines.append("    (keyboard controller or USB/HID); no code change recovers them.")
    lines.append("=" * 60)
    lines.append("")
    for (t, d, vk, make, name) in events:
        lines.append(f"{t:>8} ms  {'DOWN' if d else 'UP  '}  {name:<8} "
                     f"vk={vk:<4} make={make}")
    text = "\n".join(lines) + "\n"
    try:
        with open("raw_input_report.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("\nReport written to raw_input_report.txt")
    except Exception as e:
        print("Could not write report:", e)

def main():
    hInstance = kernel32.GetModuleHandleW(None)
    class_name = "RawInputProbeWindow"
    wndproc = WNDPROCTYPE(wnd_proc)

    wc = WNDCLASS()
    wc.lpfnWndProc = wndproc
    wc.hInstance = hInstance
    wc.lpszClassName = class_name
    if not user32.RegisterClassW(ctypes.byref(wc)):
        print("RegisterClassW failed")
        return

    hwnd = user32.CreateWindowExW(
        0, class_name, "RawInputProbe", 0, 0, 0, 0, 0,
        HWND_MESSAGE, None, hInstance, None
    )
    if not hwnd:
        print("CreateWindowExW failed")
        return

    rid = RAWINPUTDEVICE()
    rid.usUsagePage = 0x01     # generic desktop
    rid.usUsage = 0x06         # keyboard
    rid.dwFlags = RIDEV_INPUTSINK
    rid.hwndTarget = hwnd
    if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1,
                                           ctypes.sizeof(RAWINPUTDEVICE)):
        print("RegisterRawInputDevices failed")
        return

    print("=" * 60)
    print("Raw Input probe is running.")
    print("Hold the SAME keys that fail in the piano (e.g. a + f + g).")
    print("Watch 'held now' - it lists every key raw input currently sees down.")
    print("Close this window or press Ctrl+C to stop and save the report.")
    print("=" * 60)

    msg = wt.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    except KeyboardInterrupt:
        pass
    finally:
        write_report()

if __name__ == "__main__":
    main()
