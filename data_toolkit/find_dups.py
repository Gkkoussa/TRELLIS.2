import pandas as pd
import os

ROOT = "/nfs/turbo/coe-jjparkcv-medium/gpranav/objxl_4k"
csv = pd.read_csv(f"{ROOT}/asset_stats/metadata.csv")
print(csv.columns)