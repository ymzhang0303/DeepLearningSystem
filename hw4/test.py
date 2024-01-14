import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import mugrade
import torch

import needle as ndl
from needle import backend_ndarray as nd

np.random.seed(1)

print("Test 1")
shape = (2,2)
axes = None
device = ndl.cpu()
print("use cpu")
_A = np.random.randn(*shape).astype(np.float32)
_C = np.random.randn(*shape).astype(np.float32)
A = ndl.Tensor(nd.array(_A), device=device)
print("A", A)
C = ndl.Tensor(nd.array(_C), device=device)
print("C", C)
B = C - A
print("B", B)
D = 1 - A
print("D", D)
