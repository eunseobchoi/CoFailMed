"""Public primary dispatcher contract and fail-closed input checks."""
import json
import sys

import numpy as np
import pytest

from analysis import effective_size
from analysis.tests.test_effective_size import _case_ids


def test_primary_cli(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for task in effective_size.PRIMARY_TASKS:
        ids = _case_ids(task)
        for fm in ('a', 'b', 'c', 'd'):
            for seed in (42, 43):
                path = tmp_path / 'per_case_dice' / task / fm / f'seed_{seed}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(task=task, fm=fm, seed=seed, test_ids=ids,
                                               per_case_dice=rng.uniform(.3, 1, len(ids)).tolist())))
        path = tmp_path / 'stats/m_eff' / f'{task}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'mono_pools': [{'fm': fm} for fm in ('a', 'b', 'c', 'd')]}))
    monkeypatch.setattr(sys, 'argv', ['effective_size', '--traces-root', str(tmp_path), '--output-dir',
                                    str(tmp_path / 'out'), '--bootstrap-b', '2'])
    assert effective_size.main() == 0
    metrics = json.loads((tmp_path / 'out/metrics.json').read_text())
    assert [r['task'] for r in metrics['configurations']] == list(effective_size.PRIMARY_TASKS)
    assert all(r['status'] == 'complete' and r['K_bundle_count'] == 4 and r['M_trace_count'] == 8
               for r in metrics['configurations'])
    assert set(metrics['input_roots']) == {'traces_root', 'stats_root'}
    assert (tmp_path / 'out/effective_size_table.md').is_file()
    with pytest.raises(ValueError, match='unknown primary task'):
        effective_size.load_released_task('msd_spleen')
    (tmp_path / 'stats/m_eff/kvasir.json').unlink()
    with pytest.raises(FileNotFoundError):
        effective_size.build_metrics(b=2, seed=0)


def test_cli_requires_root(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['effective_size'])
    with pytest.raises(SystemExit) as error:
        effective_size.main()
    assert error.value.code == 2
