import os
from pathlib import Path

import numpy as np
import pandas as pd

latent_root = Path(os.environ["LATENT_ROOT"])
rows = []

for npz_path in sorted(latent_root.glob("*.npz")):
    cache_path = latent_root / f"{npz_path.stem}.cache.pt"
    if not cache_path.exists():
        continue

    with np.load(npz_path) as data:
        rows.append({
            "sha256": npz_path.stem,
            "gaussian_distance_latent_encoded": True,
            "gaussian_distance_latent_tokens": int(data["coords"].shape[0]),
        })

df = pd.DataFrame(rows)
out = latent_root / "metadata.csv"
df.to_csv(out, index=False)
print(f"wrote {len(df)} rows to {out}")