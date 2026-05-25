import random
import numpy as np
import torch

""" Model name """
model_name = "distilbert-base-uncased"


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Use GPU if available
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
