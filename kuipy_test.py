import torch
import kuipy 
import time

A = torch.randn(128, 128, device="cuda", dtype=torch.float32)
B = torch.randn(128, 128, device="cuda", dtype=torch.float32)
with kuipy.KuiperMode():
    t0 = time.time()
    C = torch.mm(A, B)
    t1 = time.time()
    print(t1 - t0)
    C2 = torch.mm(C, B)
    print(time.time() - t1)
