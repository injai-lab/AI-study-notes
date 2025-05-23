import pandas as pd
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

df = pd.read_csv("data/train.csv")
print(df.head())