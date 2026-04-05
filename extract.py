import sys
import torch
from main import LLNCAConfig

state = torch.load(sys.argv[1], weights_only=False)
torch.save(state["nca"], sys.argv[2])
