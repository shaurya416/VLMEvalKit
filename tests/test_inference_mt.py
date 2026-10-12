"""Tests for resuming multi-turn inference in ``vlmeval/inference_mt.py``.

The module is loaded from its source file with ``vlmeval``, torch and the model registry stubbed,
so these tests run without the model dependencies the full package imports.
"""
import importlib.util
import pickle
import sys
import types

import pandas as pd
import pytest

FAIL_MSG = 'Failed to obtain answer via API.'


def _dump(obj, path):
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)


def _load(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)


class _Tqdm:
    def __init__(self, iterable=(), **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)


@pytest.fixture
def inference_mt(monkeypatch):
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__path__ = []
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    stub('vlmeval')
    stub('vlmeval.config', supported_VLM={})
    stub('vlmeval.smp', dump=_dump, load=_load, get_pred_file_path=None,
         get_rank_and_world_size=lambda: (0, 1),
         resolve_dataset_alias_name=lambda name, alias=None: alias or name)
    stub('vlmeval.utils')
    torch = stub('torch', cuda=types.SimpleNamespace(empty_cache=lambda: None))
    torch.distributed = stub('torch.distributed')
    stub('tqdm', tqdm=_Tqdm)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    mp_util = load('vlmeval.utils.mp_util', 'vlmeval/utils/mp_util.py')
    sys.modules['vlmeval.utils'].track_progress_rich = mp_util.track_progress_rich
    return load('vlmeval.inference_mt', 'vlmeval/inference_mt.py')


class _Dataset:
    dataset_name = 'MMDU'

    def __init__(self):
        self.data = pd.DataFrame({'index': [0, 1]})

    def build_prompt(self, line):
        turns = []
        for i in range(2):
            turns.append(dict(role='user', content=f'question {i}'))
            turns.append(dict(role='assistant', content=f'reference {i}'))
        return turns


class _ChatModel:
    is_api = True

    def __init__(self):
        self.calls = 0

    def chat_inner(self, inputs, **kwargs):
        raise NotImplementedError

    def chat(self, messages, dataset=None):
        self.calls += 1
        return f'answer to turn {len(messages) // 2 + 1}'


@pytest.mark.parametrize('failed_turn', [
    # BaseAPI.chat once every conversation turn has been dropped.
    FAIL_MSG + ': Failed with all possible conversation turns.',
    # chat_mt when model.chat raises.
    FAIL_MSG + 'Connection reset by peer',
])
def test_resumed_run_retries_samples_with_a_failed_turn(inference_mt, tmp_path, failed_turn):
    _dump({'0': ['first answer', failed_turn], '1': ['kept', 'kept']},
          tmp_path / 'Model_MMDU_checkpoint.pkl')
    model = _ChatModel()

    result = inference_mt.infer_data_api(model, str(tmp_path), 'Model', _Dataset())

    assert result == {0: ['answer to turn 1', 'answer to turn 2'], 1: ['kept', 'kept']}
    assert model.calls == 2


def test_keep_failed_reuses_samples_with_a_failed_turn(inference_mt, tmp_path):
    failed = ['first answer', FAIL_MSG + 'Connection reset by peer']
    _dump({'0': failed, '1': ['kept', 'kept']}, tmp_path / 'Model_MMDU_checkpoint.pkl')
    model = _ChatModel()

    result = inference_mt.infer_data_api(model, str(tmp_path), 'Model', _Dataset(),
                                         retry_failed=False)

    assert result == {0: failed, 1: ['kept', 'kept']}
    assert model.calls == 0
