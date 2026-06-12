"""Windows-only debug helper: print RAM and commit-charge availability."""

import ctypes
import struct


class M(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("x", ctypes.c_ulonglong)]


m = M()
m.dwLength = ctypes.sizeof(M)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
g = 1024 ** 3
print(struct.calcsize("P") * 8, "bit python")
print(f"RAM total {m.ullTotalPhys / g:.1f} GB, avail {m.ullAvailPhys / g:.1f} GB")
print(f"commit limit {m.ullTotalPageFile / g:.1f} GB, avail {m.ullAvailPageFile / g:.1f} GB")
