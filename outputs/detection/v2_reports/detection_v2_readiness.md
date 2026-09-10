# Detection V2 (CPU) - Phase 22 - Readiness Check

- [x] Stage-1 preprocessing unchanged
- [x] 14-channel ordering unchanged
- [x] train/val/test split unchanged
- [x] no test leakage (LOCKED marker required for test eval)
- [x] CPU training completed (GPU version deferred)
- [x] fair loss comparison completed (same protocol per arm)
- [x] best model selected using validation
- [x] threshold selected using validation
- [x] test evaluated only after freezing
- [x] error analysis completed
- [x] prediction visualizations generated
- [x] checkpoint saved (models/detection/detection_v2_best.pth)
- [x] inference module (`src/detection/inference.py`) loads V2 arch
- [x] model metadata saved (detection_v2_metadata.json)
- [x] mask-to-polygon pipeline tested (pixel coords; CRS documented)
- [x] frontend-ready output schema documented in polygon GeoJSON
- [x] Monitoring-ready output: mask + probability + polygon + confidence
