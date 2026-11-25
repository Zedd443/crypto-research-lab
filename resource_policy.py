"""Conservative worker budget; actual workload measurements override estimates."""
import os
import psutil

GIB=1024**3

def budget(requested=4,per_worker_bytes=768*1024**2,reserve_bytes=GIB,threads_per_worker=1):
    if min(requested,per_worker_bytes,threads_per_worker)<1 or reserve_bytes<0:
        raise ValueError("Invalid resource budget")
    available=psutil.virtual_memory().available
    cpu=os.cpu_count() or 1
    mem_limit=max(0,(available-reserve_bytes)//per_worker_bytes)
    if mem_limit<1:
        raise MemoryError("Not enough estimated free RAM for one worker")
    return max(1,min(requested,cpu//threads_per_worker or 1,int(mem_limit)))

def tree_rss():
    p=psutil.Process()
    processes=[p,*p.children(recursive=True)]
    total=0
    for child in processes:
        try:total+=child.memory_info().rss
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return total
