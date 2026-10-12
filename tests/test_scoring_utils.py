"""Tests for scoring and dataset-parsing helpers.

Each module is loaded from its source file with ``vlmeval`` and its heavy third-party imports
stubbed, so these tests run without the model dependencies the full package imports.
"""
import importlib.util
import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _noop(*args, **kwargs):
    return None


class _ParsedLatex:
    # Stands in for a parsed LaTeX expression whose text form is the model's own answer.
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


@pytest.fixture
def stub(monkeypatch):
    def install(name, **attrs):
        module = types.ModuleType(name)
        module.__path__ = []
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module
    return install


@pytest.fixture
def stubbed_vlmeval(stub):
    stub('vlmeval')
    stub('vlmeval.smp', load=_noop, get_logger=logging.getLogger)
    stub('vlmeval.smp.file', load=_noop)
    stub('vlmeval.utils', can_infer=_noop)
    stub('vlmeval.dataset')
    stub('vlmeval.dataset.utils')
    stub('vlmeval.dataset.utils.multiple_choice', extract_answer_from_item=_noop)
    stub('vlmeval.dataset.EgoExoBench')
    stub('torch')
    stub('torchvision')
    stub('PIL', Image=types.SimpleNamespace(BILINEAR=2), ImageOps=types.SimpleNamespace())
    stub('timeout_decorator', timeout=lambda *args, **kwargs: (lambda func: func))
    stub('latex2sympy2_extended', latex2sympy=_ParsedLatex)


@pytest.fixture
def probe(stub):
    # A prediction or dataset field that runs code would record a hit here.
    return stub('vlmeval_probe', hits=[])


RATING_MODULES = [
    ('vlmeval.dataset.utils.mvbench', 'vlmeval/dataset/utils/mvbench.py', 'task_type'),
    ('vlmeval.dataset.utils.tamperbench', 'vlmeval/dataset/utils/tamperbench.py', 'task_type'),
    ('vlmeval.dataset.EgoExoBench.utils', 'vlmeval/dataset/EgoExoBench/utils.py', 'subtask_type'),
]


@pytest.mark.parametrize('name, path, category', RATING_MODULES)
def test_dimension_rating_does_not_count_failures_as_correct(stubbed_vlmeval, name, path,
                                                             category):
    module = _load(name, path)
    # 1 correct, 1 wrong, 1 failed API call or judge (-1), 1 missing prediction (NaN).
    scores = pd.DataFrame({category: ['task'] * 4, 'score': [1.0, 0.0, -1.0, np.nan]})
    module.load = lambda path: scores
    rating = module.get_dimension_rating('score_file')
    assert rating['task'] == [1, 4, '25.00%']
    assert rating['overall'] == [1, 4, '25.00%']


EQUALITY_MODULES = [
    ('vlmeval.dataset.utils.mathv', 'vlmeval/dataset/utils/mathv.py'),
    ('vlmeval.dataset.utils.lens', 'vlmeval/dataset/utils/lens.py'),
]


@pytest.mark.parametrize('name, path', EQUALITY_MODULES)
def test_is_equal_does_not_execute_prediction(stubbed_vlmeval, probe, name, path):
    module = _load(name, path)
    prediction = "__import__('vlmeval_probe').hits.append(1) or 1"
    assert module.is_equal(prediction, '1') is False
    assert probe.hits == []


@pytest.mark.parametrize('name, path', EQUALITY_MODULES)
def test_is_equal_still_compares_numbers(stubbed_vlmeval, name, path):
    module = _load(name, path)
    assert module.is_equal('3/11', '0.272727272') is True
    assert module.is_equal('1e3', '1000') is True
    assert module.is_equal('-2', '-2.0') is True
    assert module.is_equal('2**3', '8') is True
    assert module.is_equal('7', '8') is False
    assert module.is_equal('[2007, 2008]', '[2007,2008]') is False


@pytest.fixture
def misc(stub):
    stub('huggingface_hub')
    stub('huggingface_hub.utils')
    stub('huggingface_hub.utils._cache_manager', _scan_cached_repo=_noop)
    stub('sty', fg=object())
    return _load('_misc_under_test', 'vlmeval/smp/misc.py')


def test_toliststr_does_not_execute_dataset_field(misc, probe):
    with pytest.raises(ValueError):
        misc.toliststr("[__import__('vlmeval_probe').hits.append(1)]")
    assert probe.hits == []


def test_istype_does_not_execute_dataset_field(misc, probe):
    assert misc.istype("__import__('vlmeval_probe').hits.append(1) or [1]", list) is False
    assert probe.hits == []


def test_list_fields_still_parsed(misc):
    assert misc.toliststr("['a.jpg', 'b.jpg']") == ['a.jpg', 'b.jpg']
    assert misc.toliststr('a.jpg') == ['a.jpg']
    assert misc.istype("['cat', 'dog']", list) is True
    assert misc.istype('7', int) is True
    assert misc.istype('cat', list) is False


@pytest.fixture
def mme_reasoning(stubbed_vlmeval, stub):
    stub('sympy')
    return _load('vlmeval.dataset.utils.mme_reasoning', 'vlmeval/dataset/utils/mme_reasoning.py')


def test_24points_accepts_solutions_with_fractional_steps(mme_reasoning):
    judge = mme_reasoning.judge_24points_function
    # In floating point both formulas miss 24 (23.99999999999999 and 24.000000000000007).
    assert judge('8/(3-8/3)=24', None, {'init_cards': [3, 3, 8, 8]}) is True
    assert judge('4/(1-5/6)=24', None, {'init_cards': [1, 4, 5, 6]}) is True


def test_24points_still_checks_value_and_cards(mme_reasoning):
    judge = mme_reasoning.judge_24points_function
    assert judge('(13-9-1)*8=24', None, {'init_cards': [1, 8, 9, 13]}) is True
    assert judge('9/3*1*8=24', None, {'init_cards': [1, 3, 8, 9]}) is True
    assert judge('\t(13-9-1)*8=24\n', None, {'init_cards': [1, 8, 9, 13]}) is True
    assert judge('(13-9)*8=24', None, {'init_cards': [8, 9, 13]}) is False
    assert judge('4*6=24', None, {'init_cards': [4, 6, 1, 1]}) is False
    assert judge('8/(3-3)=24', None, {'init_cards': [3, 3, 8]}) is False


def test_24points_does_not_execute_response(mme_reasoning, probe):
    # Spaces are stripped before evaluation, so the payload is written without any.
    response = "__import__('vlmeval_probe').hits.append(1)or(24)=24"
    assert mme_reasoning.judge_24points_function(response, None, {'init_cards': [1, 24]}) is False
    assert probe.hits == []


@pytest.fixture
def mmlongbenchdoc(stubbed_vlmeval, stub):
    stub('torchvision', transforms=types.SimpleNamespace())
    stub('torchvision.transforms')
    stub('PIL', Image=types.SimpleNamespace(), ImageDraw=types.SimpleNamespace(),
         ImageFont=types.SimpleNamespace())
    stub('vlmeval.smp', decode_base64_to_image_file=_noop, dump=_noop,
         encode_image_to_base64=_noop, get_logger=logging.getLogger, istype=_noop,
         listinstr=_noop, load=_noop, process_punctuation=_noop, read_ok=_noop, toliststr=_noop)
    vqa_eval = _load('_vqa_eval_under_test', 'vlmeval/dataset/utils/vqa_eval.py')
    stub('vlmeval.dataset.utils', build_judge=_noop,
         levenshtein_distance=vqa_eval.levenshtein_distance)
    stub('vlmeval.dataset.utils.judge_cache', get_judge_cache_file=_noop,
         get_judge_detail_file=_noop, get_judge_named_legacy_cache_file=_noop,
         get_judge_score_file=_noop, load_judge_cache=_noop, run_cached_tasks=_noop)
    stub('vlmeval.dataset.image_base', ImageBaseDataset=object)
    return _load('vlmeval.dataset.mmlongbenchdoc', 'vlmeval/dataset/mmlongbenchdoc.py')


def _list_score(module, gt, pred):
    # MMLongBench_acc scores an answer that eval_score cannot parse as 0.
    try:
        return module.eval_score(gt, pred, 'List')
    except Exception:
        return 0.0


def test_mmlongbench_list_answer_does_not_execute_prediction(mmlongbenchdoc, probe):
    # The extracted answer is the judge's copy of the model's own answer.
    pred = "[__import__('vlmeval_probe').hits.append(1), 'Asia']"
    assert _list_score(mmlongbenchdoc, "['Europe', 'Asia']", pred) == 0.0
    assert probe.hits == []


def test_mmlongbench_list_answers_still_scored(mmlongbenchdoc):
    assert _list_score(mmlongbenchdoc, "['Europe', 'Asia']", "['asia', 'europe']") == 1.0
    assert _list_score(mmlongbenchdoc, "['2019', '2020']", "[2020, 2019]") == 1.0
    assert _list_score(mmlongbenchdoc, "['Europe', 'Asia']", "['Asia']") == 0.0
    assert _list_score(mmlongbenchdoc, "['Europe', 'Asia']", "['Europe', 'Asia'") == 0.0
