"""Uncached reader copied from the existing RGB full-data run; no training code."""
import ctypes as c
from ctypes import wintypes as w
import os
from pathlib import Path

class UncachedReader:
    """VirtualAlloc-aligned FILE_FLAG_NO_BUFFERING reads, no OS data cache."""
    def __init__(self, path):
        if os.name != 'nt': raise RuntimeError('this experiment explicitly requires Windows direct I/O')
        self.k = c.WinDLL('kernel32', use_last_error=True)
        self.k.CreateFileW.argtypes = [w.LPCWSTR,w.DWORD,w.DWORD,c.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
        self.k.CreateFileW.restype = w.HANDLE
        self.k.VirtualAlloc.argtypes = [c.c_void_p,c.c_size_t,w.DWORD,w.DWORD]
        self.k.VirtualAlloc.restype = c.c_void_p
        self.k.VirtualFree.argtypes = [c.c_void_p,c.c_size_t,w.DWORD]
        self.k.ReadFile.argtypes = [w.HANDLE,c.c_void_p,w.DWORD,c.POINTER(w.DWORD),c.c_void_p]
        self.k.SetFilePointerEx.argtypes = [w.HANDLE,c.c_longlong,c.c_void_p,w.DWORD]
        self.k.CloseHandle.argtypes = [w.HANDLE]
        # GetFileInformationByHandleEx(FileStorageInfo): query physical/logical sectors.
        self.k.GetFileInformationByHandleEx.argtypes = [w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
        self.handle = self.k.CreateFileW(str(Path(path).resolve()),0x80000000,1,None,3,0x20000000|0x10000000,None)
        if self.handle == c.c_void_p(-1).value: raise c.WinError(c.get_last_error())
        info = (w.DWORD * 7)()
        if not self.k.GetFileInformationByHandleEx(self.handle,16,c.byref(info),c.sizeof(info)):
            self.close(); raise c.WinError(c.get_last_error())
        self.alignment = max(4096, int(info[0]), int(info[1]), int(info[2]))
        if self.alignment > 65536 or self.alignment & (self.alignment-1):
            self.close(); raise ValueError('unsupported disk alignment')
        self.pointer = None; self.capacity = 0
    def read(self, offset, size):
        start = offset // self.alignment * self.alignment
        delta = offset - start
        amount = (delta + size + self.alignment - 1) // self.alignment * self.alignment
        if amount > self.capacity:
            if self.pointer: self.k.VirtualFree(self.pointer,0,0x8000)
            self.capacity = max(amount,2**20)
            self.pointer = self.k.VirtualAlloc(None,self.capacity,0x1000|0x2000,4)
            if not self.pointer: raise c.WinError(c.get_last_error())
        if not self.k.SetFilePointerEx(self.handle,start,None,0): raise c.WinError(c.get_last_error())
        actual = w.DWORD()
        if not self.k.ReadFile(self.handle,self.pointer,amount,c.byref(actual),None): raise c.WinError(c.get_last_error())
        if actual.value < delta + size: raise OSError('truncated direct JPEG read')
        return c.string_at(self.pointer+delta,size)
    def close(self):
        if getattr(self,'pointer',None): self.k.VirtualFree(self.pointer,0,0x8000); self.pointer=None
        handle = getattr(self,'handle',None)
        if handle is not None and handle != -1 and handle != 18446744073709551615:
            self.k.CloseHandle(handle); self.handle=None
    def __del__(self): self.close()
