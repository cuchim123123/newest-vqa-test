"""Check what's stored inside glove embedding cache files."""
import torch

for size in [2717, 4203]:
    path = f"data/glove_cache/emb_{size}_300.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== {path} ===")
    if isinstance(data, dict):
        print(f"  Type: dict, keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: Tensor shape={v.shape}")
            elif isinstance(v, list):
                print(f"    {k}: list len={len(v)}, first 5: {v[:5]}")
            elif isinstance(v, dict):
                items = list(v.items())[:5]
                print(f"    {k}: dict len={len(v)}, first 5: {items}")
            else:
                print(f"    {k}: {type(v).__name__} = {v}")
    elif isinstance(data, torch.Tensor):
        print(f"  Type: Tensor, shape={data.shape}")
    else:
        print(f"  Type: {type(data).__name__}")
