"""Environment check: run this on a GPU node before submitting anything.

Confirms torch sees the GPU and that the build actually supports its compute
capability (RTX Pro 6000 is sm_120; older central modules are built for sm_70/sm_80
and will fail here rather than mid-training).
"""
import torch

print(f"torch {torch.__version__}, torchvision available: ", end="")
try:
    import torchvision
    print(torchvision.__version__)
except ImportError:
    print("NO")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version torch was built with: {torch.version.cuda}")
print(f"Architectures this build supports: {torch.cuda.get_arch_list()}")

if not torch.cuda.is_available():
    raise SystemExit("No GPU visible. Are you on a GPU node / did you request --gres=gpu:1?")

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {name}, compute capability {cap[0]}.{cap[1]}, {mem:.0f} GB")

arch = f"sm_{cap[0]}{cap[1]}"
if arch not in [a.split("_compute")[0] for a in torch.cuda.get_arch_list()]:
    print(f"WARNING: {arch} not in this build's arch list; kernels may fail to launch.")

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
print(f"bf16 matmul on GPU OK (result finite: {y == y})")
print("Environment check passed.")
