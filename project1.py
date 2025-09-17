import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

path = Path.cwd() / 'synthetic_bond_dataset.csv'
df = pd.read_csv(path)

print(df.head())
print(df.shape)
print(df.dtyps)