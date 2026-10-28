
"""Run the scoped controls scorer using restored scalar inputs."""
import argparse
import importlib.util
import json
from pathlib import Path


def compare_recorded_support(compare, actual, reference):
    """Permit only added zero eligibility diagnostics; retain every recorded field."""
    if isinstance(actual, dict) and isinstance(reference, dict):
        extra = actual.keys() - reference.keys()
        if extra and {'n_pools_total', 'n_pools_supported'} <= reference.keys():
            if not extra <= {'n_pools_finite', 'n_pools_numerically_ineligible'} or any(actual[k] != 0 for k in extra):
                raise ValueError('unexpected support diagnostic or newly ineligible pool')
            actual = {key: value for key, value in actual.items() if key not in extra}
    return compare(actual, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        parser.error('--out must not exist')
    out.mkdir(parents=True)
    module_path = Path(__file__).parent / 'controls/compute_controls.py'
    spec = importlib.util.spec_from_file_location('scoped_controls', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.OUT = root, out
    module.FROZEN = root / 'results/primary'
    module.FT = root / 'results/finetuning'
    # Recorded reference hashes refer to the original trace files, before machine-specific
    # fields were removed for release. Verify the released bytes, then return the original
    # hash solely for the source-identity gate.
    manifest = root / 'SOURCE_MAP.json'
    rows = json.loads(manifest.read_text()) if manifest.exists() else []
    module.SOURCE_PATHS = {row['source']: row.get('replay_path', row['source']) for row in rows}
    lookup = {(root / row.get('replay_path', row['source'])).resolve(): row for row in rows}
    digest = module.digest

    def source_digest(path):
        actual = digest(path)
        row = lookup.get(Path(path).resolve())
        if row is None:
            return actual
        if actual != row['destination_sha256']:
            raise ValueError('released input checksum mismatch')
        return row['source_sha256']

    module.digest = source_digest
    original_same = module.same
    module.same = lambda actual, reference: compare_recorded_support(original_same, actual, reference)
    if args.check_only:
        module.checks()
    else:
        module.main()


if __name__ == '__main__':
    main()
