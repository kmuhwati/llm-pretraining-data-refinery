"""Materialise the whole refinery asset graph in-process, then build the report.

    python scripts/run_pipeline.py            # full run
    python scripts/run_pipeline.py --no-report
"""
import os
import sys
import time
from pathlib import Path

# Worker processes inherit these. Each worker is single-threaded by design; letting
# OpenBLAS reserve a thread pool per process exhausts memory with 8+ workers.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dagster import materialize

    from refinery import config as C
    from refinery.assets import ALL_ASSETS

    for d in (C.STAGES, C.OUTPUT, C.FIGURES, C.TABLES):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = materialize(ALL_ASSETS)
    print(f"pipeline {'succeeded' if result.success else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)
    if result.success and "--no-report" not in sys.argv:
        from refinery.report import build_report
        build_report()
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
