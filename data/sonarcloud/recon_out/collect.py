import numpy as np, ast

raw_path = "/netscratch/rsonawane/ex_ai/OpenPCDet/data/sonarcloud/recon_out/boat_noTerrain_ori1.npy"
npz_path = "/netscratch/rsonawane/ex_ai/OpenPCDet/sota_typeA_explanations_boat/boat_noTerrain_ori1/det001_Pedestrian_rank2/typeA_explanation.npz"

xyz_scale = 5.0
xyz_shift = np.array([25.0, 0.0, -1.5], dtype=np.float32)

raw = np.load(raw_path).astype(np.float32)
xform = raw * xyz_scale + xyz_shift

print("\n[RAW]")
print(" min", raw.min(0), "max", raw.max(0), "mean", raw.mean(0))

print("\n[XFORM]")
print(" min", xform.min(0), "max", xform.max(0), "mean", xform.mean(0))

d = np.load(npz_path, allow_pickle=True)
diag_str = d["diag"].item()
if isinstance(diag_str, (bytes, bytearray)):
    diag_str = diag_str.decode("utf-8")
diag = ast.literal_eval(diag_str)

print("\n[DIAG]")
print(" orig_scalar:", diag["orig_scalar"])
print(" orig_region_hits:", diag["orig_region_hits"])
