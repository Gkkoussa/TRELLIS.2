
import os, pandas as pd
root = os.environ["ROOT"]
stage = os.path.join(root, "mesh_dumps")
df = pd.read_csv(os.path.join(stage, "metadata.csv"))
print("mesh_dumped:", int(df["mesh_dumped"].sum()))
print("pickle_files:", len([f for f in os.listdir(stage) if f.endswith(".pickle")]))