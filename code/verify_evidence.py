"""Verify the repository evidence snapshot without external dependencies."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'evidence_manifest.json').read_text(encoding='utf-8'))
manifest.update(json.loads((root / 'source_results_manifest.json').read_text(encoding='utf-8')))
failures = []
for name, expected in manifest.items():
    path = root / name
    if not path.is_file():
        failures.append(f'Missing: {name}')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(f'Checksum mismatch: {name}')
if failures:
    print('\n'.join(failures))
    sys.exit(1)
print(f'All {len(manifest)} evidence, source, and result checksums match.')
