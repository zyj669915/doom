"""只读核对历史checkpoint的源码清单，不加载tensor或脑图。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'experiment/runs_next/20261020-210028-train-A6-301910d4/checkpoint-000186871-e92d97ec/manifest.json'


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    expected = manifest['provenance']['sources']
    differences = []
    for relative, digest in expected.items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != digest:
            differences.append(relative)
    result = {'manifest': str(MANIFEST), 'files_checked': len(expected),
              'differences': differences, 'unchanged': not differences,
              'scope': 'historical source manifest only; graph not rehashed'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not differences else 1


if __name__ == '__main__':
    raise SystemExit(main())
