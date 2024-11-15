# Script to clear CUDA memory in PyTorch

import torch

# Clear CUDA memory
torch.cuda.empty_cache()

print("Cleared CUDA memory cache.")
