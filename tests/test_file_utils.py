"""Tests for the dump / download helpers in ``vlmeval/smp/file.py``.

The module is loaded from its source file with its package siblings and third-party imports
stubbed, so these tests run without the model dependencies the full package imports.
"""
import importlib.util
import json
import logging
import os
import sys
import types
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest


class _ProgressBar:
    def __init__(self, *args, **kwargs):
        self.n = 0
        self.total = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n=1):
        self.n += n


@pytest.fixture
def file_utils(monkeypatch):
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__path__ = []
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    stub('vlmeval')
    stub('vlmeval.smp')
    stub('vlmeval.smp.log', get_logger=logging.getLogger)
    stub('vlmeval.smp.misc', toliststr=lambda s: [s])
    stub('vlmeval.smp.vlm', decode_base64_to_image_file=lambda *args: None)
    stub('validators', url=lambda s: False)
    stub('tqdm', tqdm=_ProgressBar)
    spec = importlib.util.spec_from_file_location('vlmeval.smp.file', 'vlmeval/smp/file.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'vlmeval.smp.file', module)
    spec.loader.exec_module(module)
    return module


def test_dump_json_encodes_numpy_values(file_utils, tmp_path):
    target = tmp_path / 'acc.json'
    file_utils.dump({'Overall': np.float32(0.5), 'hit': np.bool_(True), 'counts': np.arange(2)},
                    str(target))
    assert json.loads(target.read_text()) == {'Overall': 0.5, 'hit': True, 'counts': [0, 1]}
    assert not (tmp_path / 'acc.pkl').exists()


def _predictions(rows=5000, bad_row=3000):
    # A lone surrogate (half of an emoji) in one model answer cannot be encoded as UTF-8.
    preds = [f'answer {i}' for i in range(rows)]
    preds[bad_row] = 'broken emoji \ud83d'
    return pd.DataFrame({'index': range(rows), 'prediction': preds})


@pytest.mark.parametrize('ext', ['tsv', 'csv', 'json'])
def test_failed_dump_leaves_no_truncated_file_behind(file_utils, tmp_path, ext):
    data = _predictions()
    target = tmp_path / f'Model_Bench.{ext}'
    file_utils.dump(data, str(target))
    # The write fails part-way; a truncated table at the target would load as a shorter result.
    assert sorted(os.listdir(tmp_path)) == ['Model_Bench.pkl']
    assert len(file_utils.load(str(tmp_path / 'Model_Bench.pkl'))) == len(data)


def test_failed_dump_removes_the_earlier_result(file_utils, tmp_path):
    target = tmp_path / 'Model_Bench.tsv'
    file_utils.dump(pd.DataFrame({'index': [0], 'prediction': ['stale']}), str(target))
    file_utils.dump(_predictions(), str(target))
    assert sorted(os.listdir(tmp_path)) == ['Model_Bench.pkl']


def test_interrupted_dump_keeps_the_previous_file(file_utils, tmp_path, monkeypatch):
    target = tmp_path / 'Model_Bench_checkpoint.pkl'
    file_utils.dump({'0': 'first answer'}, str(target))

    def interrupted(data, stream):
        stream.write(b'\x80\x04')
        raise KeyboardInterrupt

    monkeypatch.setattr(file_utils.pickle, 'dump', interrupted)
    with pytest.raises(KeyboardInterrupt):
        file_utils.dump({'0': 'first answer', '1': 'second answer'}, str(target))
    assert os.listdir(tmp_path) == ['Model_Bench_checkpoint.pkl']
    assert file_utils.load(str(target)) == {'0': 'first answer'}


def test_unpicklable_checkpoint_keeps_the_previous_file(file_utils, tmp_path):
    target = tmp_path / 'Model_Bench_checkpoint.pkl'
    file_utils.dump({'0': 'first answer'}, str(target))
    with pytest.raises(Exception):
        file_utils.dump({'0': 'first answer', '1': lambda: None}, str(target))
    assert os.listdir(tmp_path) == ['Model_Bench_checkpoint.pkl']
    assert file_utils.load(str(target)) == {'0': 'first answer'}


def test_dump_writes_through_a_symlinked_target(file_utils, tmp_path):
    run_file = tmp_path / 'run' / 'Model_Bench.tsv'
    run_file.parent.mkdir()
    link = tmp_path / 'Model_Bench.tsv'
    file_utils.dump(pd.DataFrame({'index': [0], 'prediction': ['old']}), str(run_file))
    link.symlink_to(run_file)
    file_utils.dump(pd.DataFrame({'index': [0], 'prediction': ['new']}), str(link))
    assert link.is_symlink()
    assert list(file_utils.load(str(run_file))['prediction']) == ['new']


def _write_then_fail(url, filename=None, reporthook=None):
    with open(filename, 'w') as fout:
        fout.write('index\tquestion\n0\tWhat is in')
    raise urllib.error.ContentTooShortError('retrieval incomplete', None)


def test_failed_download_leaves_no_file_behind(file_utils, tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, 'urlretrieve', _write_then_fail)
    target = tmp_path / 'Bench.tsv'
    with pytest.raises(Exception, match='Failed to download'):
        file_utils.download_file('https://example.com/Bench.tsv', str(target))
    # load() and prepare_tsv() reuse any file already at the target as a finished download.
    assert os.listdir(tmp_path) == []


def test_interrupted_download_leaves_no_file_behind(file_utils, tmp_path, monkeypatch):
    def retrieve(url, filename=None, reporthook=None):
        with open(filename, 'w') as fout:
            fout.write('index\tquestion\n0\tWhat is in')
        raise KeyboardInterrupt

    monkeypatch.setattr(urllib.request, 'urlretrieve', retrieve)
    with pytest.raises(KeyboardInterrupt):
        file_utils.download_file('https://example.com/Bench.tsv', str(tmp_path / 'Bench.tsv'))
    assert os.listdir(tmp_path) == []


def test_completed_download_is_moved_into_place(file_utils, tmp_path, monkeypatch):
    def retrieve(url, filename=None, reporthook=None):
        with open(filename, 'w') as fout:
            fout.write('index\tquestion\n0\tWhat is in the image?\n')

    monkeypatch.setattr(urllib.request, 'urlretrieve', retrieve)
    target = tmp_path / 'Bench.tsv'
    assert file_utils.download_file('https://example.com/Bench.tsv', str(target)) == str(target)
    assert target.read_text() == 'index\tquestion\n0\tWhat is in the image?\n'
    assert os.listdir(tmp_path) == ['Bench.tsv']


def test_mirror_fallback_still_completes_the_download(file_utils, tmp_path, monkeypatch):
    def retrieve(url, filename=None, reporthook=None):
        if 'huggingface.co' in url:
            _write_then_fail(url, filename)
        with open(filename, 'w') as fout:
            fout.write('from the mirror\n')

    monkeypatch.setattr(urllib.request, 'urlretrieve', retrieve)
    target = tmp_path / 'Bench.tsv'
    file_utils.download_file('https://huggingface.co/datasets/x/Bench.tsv', str(target))
    assert target.read_text() == 'from the mirror\n'
    assert os.listdir(tmp_path) == ['Bench.tsv']
