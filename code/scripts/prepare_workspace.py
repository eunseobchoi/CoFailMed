#!/usr/bin/env python3
"""Verify saved inputs and table generators, then prepare an analysis workspace."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--include-phikon', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must not exist')
    artifacts, output = args.artifacts.resolve(), args.out.resolve()
    metadata = (artifacts / 'SOURCE_MAP.json').read_bytes()
    rows = json.loads(metadata)
    selected, targets = [], {output / 'SOURCE_MAP.json'}

    def queue(source, target, allowed_root, expected):
        source, target = source.resolve(), target.resolve()
        if not source.is_relative_to(allowed_root.resolve()) or not target.is_relative_to(output):
            raise ValueError('unsafe source map path')
        if target in targets:
            raise ValueError('duplicate workspace target')
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f'input checksum mismatch: {source.name}')
        targets.add(target)
        selected.append((source, target, expected))

    for row in rows:
        if row['destination'].startswith('phikon/') and not args.include_phikon:
            continue
        if row['kind'] not in ('json', 'exact', 'license'):
            raise ValueError('artifact map must contain data, not executable code')
        target = (output / 'PHIKON_LICENSE.txt' if row['kind'] == 'license'
                  else output / row.get('replay_path', row['source']))
        source = artifacts / row['destination']
        allowed_suffix = {'json': '.json', 'exact': '.csv', 'license': '.txt'}[row['kind']]
        if source.suffix.lower() != allowed_suffix or target.suffix.lower() != allowed_suffix:
            raise ValueError('artifact data cannot create executable or mismatched file types')
        queue(artifacts / row['destination'], target, artifacts, row['destination_sha256'])

    repository = Path(__file__).resolve().parents[2]
    generators = repository / 'code/table_generation'
    code_map = json.loads((repository / 'code/SCORER_SOURCE_MAP.json').read_text())
    generator_sources = set()
    for row in code_map:
        if row.get('source_kind') != 'table_generation':
            continue
        source = (repository / row['destination']).resolve()
        if not source.is_relative_to(generators.resolve()) or source.suffix != '.py':
            raise ValueError('unsafe table generator path')
        target = output / 'code/scripts' / source.relative_to(generators.resolve())
        queue(source, target, generators, row['destination_sha256'])
        generator_sources.add(source)
    actual_generators = {path.resolve() for path in generators.rglob("*.py")}
    if not generator_sources or generator_sources != actual_generators:
        raise ValueError("table generation code and its source map do not match")

    output.mkdir(parents=True)
    for source, target, expected in selected:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'input changed during copying: {source.name}')
    (output / 'SOURCE_MAP.json').write_bytes(metadata)
    print(f'prepared {len(selected)} files, including {len(generator_sources)} table generators')


if __name__ == '__main__':
    main()
