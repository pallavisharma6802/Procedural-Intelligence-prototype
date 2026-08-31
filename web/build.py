"""Bake the current runs into web/standalone.html (server-free) and web/artifact.html (no doc shell).

    ./.venv/bin/python web/build.py [case_id ...]
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi.casefile import CaseFile  # noqa: E402
from pi.webexport import export_case  # noqa: E402

WEB = Path(__file__).resolve().parent
DEFAULT_CASES = [
    "case01_lapchole", "case04_cath_pci", "case01_uk", "case03_trauma_exlap",
    "mmor_007_TKA", "mmor_007_audio", "case02_tka_uneventful",
]


def main(cases):
    bundle = {}
    for cid in cases:
        try:
            bundle[cid] = export_case(CaseFile.load(cid))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {cid}: {exc}")
    (WEB / "_bundle.json").write_text(json.dumps(bundle, separators=(",", ":")))

    tpl = (WEB / "index.html").read_text(encoding="utf-8")
    standalone = tpl.replace("__BUNDLE__", (WEB / "_bundle.json").read_text(encoding="utf-8"))
    (WEB / "standalone.html").write_text(standalone, encoding="utf-8")

    m = re.search(r"<head>(.*?)</head>\s*<body>(.*?)</body>\s*</html>", standalone, re.DOTALL)
    head, body = m.group(1), m.group(2)
    keep = re.findall(r"<title>.*?</title>|<link[^>]*>|<style>.*?</style>", head, re.DOTALL)
    artifact = '<meta charset="utf-8">\n' + "\n".join(keep) + "\n" + body.strip() + "\n"
    (WEB / "artifact.html").write_text(artifact, encoding="utf-8")

    kb = lambda p: round((WEB / p).stat().st_size / 1024)
    print(f"  {len(bundle)} cases  -  bundle {kb('_bundle.json')} KB  -  "
          f"standalone {kb('standalone.html')} KB  -  artifact {kb('artifact.html')} KB")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_CASES)
