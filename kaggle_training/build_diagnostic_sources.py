"""Generate standalone stage launch artifacts from the reviewed diagnostic source."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "diagnostic.py").read_text(encoding="utf-8")
for stage in ("gpu", "data"):
    target = root / f"diagnostic_{stage}_source.py"
    target.write_text(
        f"import os\nos.environ['KAGGLE_DIAGNOSTIC_STAGE'] = {stage!r}\n" + source,
        encoding="utf-8",
    )
    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    print(target)
