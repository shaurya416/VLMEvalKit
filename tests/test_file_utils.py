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
