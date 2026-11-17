"""Smoke check for the Rust module: python test_core.py"""
import numpy as np
import terrain_core

n = 128
h = np.frombuffer(terrain_core.fbm(n, seed=1), np.float32).reshape(n, n)
assert h.min() == 0 and abs(h.max() - 1) < 1e-6, "fbm should be normalized"
assert np.array_equal(h, np.frombuffer(terrain_core.fbm(n, seed=1), np.float32).reshape(n, n)), "deterministic"

e = np.frombuffer(terrain_core.erode(h.tobytes(), n, drops=20000), np.float32).reshape(n, n)
assert np.isfinite(e).all()
diff = np.abs(e - h)
assert diff.max() > 1e-3, "erosion should carve the map"
assert abs(e.mean() - h.mean()) < 0.01, "sediment roughly conserved"
try:
    terrain_core.erode(b"bad", n); raise AssertionError("bad buffer accepted")
except ValueError:
    pass
print(f"ok: max change {diff.max():.4f}, mean shift {e.mean() - h.mean():+.5f}")
