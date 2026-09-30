"""One-off (2026-10-01): rename result folders gtm/sgtm/asgtm[-tag] to mr/smr/asmr[-tag]."""
import re
import sys
from pathlib import Path

MAP = {"asgtm": "asmr", "sgtm": "smr", "gtm": "mr"}
root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
for ds in [d for d in root.iterdir() if d.is_dir()]:
    for sub in [d for d in ds.iterdir() if d.is_dir()]:
        m = re.fullmatch(r"(asgtm|sgtm|gtm)(-.*)?", sub.name)
        if m:
            target = ds / (MAP[m.group(1)] + (m.group(2) or ""))
            if target.exists():
                raise SystemExit(f"refusing to overwrite {target}")
            sub.rename(target)
            print(f"{sub} -> {target.name}")
