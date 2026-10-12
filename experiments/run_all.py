"""Runs RQ1 to RQ4 in order."""
import time

from experiments import rq1, rq2, rq3, rq4

for name, mod in (("RQ1", rq1), ("RQ2", rq2), ("RQ3", rq3), ("RQ4", rq4)):
    t = time.time()
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    mod.main()
    print(f"{name} done in {(time.time() - t) / 60:.1f} min")
