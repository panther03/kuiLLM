import torch
import kuipy 
import time

A = torch.randn(128, 64, device="cuda")
B = torch.randn(128, 64, device="cuda")
with kuipy.KuiperMode():
    t0 = time.time()
    C = torch.add(A, B)
    t1 = time.time()
    print(t1 - t0)
    C2 = torch.add(C, B)
    print(time.time() - t1)
