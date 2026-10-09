"""Tests for multiple-choice answer extraction helpers.

The modules are loaded from their source files with the ``vlmeval`` package
stubbed, so these tests run without the model dependencies the full package
imports.
"""
import importlib.util
import logging
import sys
import types

import pytest


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stubbed_vlmeval(monkeypatch):
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = []
    monkeypatch.setitem(sys.modules, 'vlmeval', vlmeval)

    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = []
    for name in ['cn_string', 'd2df', 'dump', 'get_pred_file_format', 'istype', 'load', 'timestr']:
        setattr(smp, name, lambda *args, **kwargs: None)
    smp.get_logger = lambda name: logging.getLogger(name)
    monkeypatch.setitem(sys.modules, 'vlmeval.smp', smp)

    smp_log = types.ModuleType('vlmeval.smp.log')
    smp_log.get_logger = smp.get_logger
    monkeypatch.setitem(sys.modules, 'vlmeval.smp.log', smp_log)

    smp_vlm = types.ModuleType('vlmeval.smp.vlm')
    smp_vlm.build_option_str = lambda choices: ''
    monkeypatch.setitem(sys.modules, 'vlmeval.smp.vlm', smp_vlm)

    utils = types.ModuleType('vlmeval.utils')
    for name in ['can_infer', 'can_infer_lego', 'track_progress_rich']:
        setattr(utils, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, 'vlmeval.utils', utils)


@pytest.fixture
def matching_util(stubbed_vlmeval):
    return _load('_matching_util_under_test', 'vlmeval/utils/matching_util.py')


@pytest.fixture
def multiple_choice(stubbed_vlmeval):
    return _load('_multiple_choice_under_test', 'vlmeval/dataset/utils/multiple_choice.py')


CHOICES = {'A': 'cat', 'B': 'dog', 'C': 'bird', 'D': 'fish'}


def test_verbose_answer_still_extracted(matching_util):
    answer = "The correct answer is **B**. Here's why: A and C are wrong."
    assert matching_util.can_infer(answer, dict(CHOICES)) == 'B'
    assert matching_util.can_infer('The answer is C. Not A, not D.', dict(CHOICES)) == 'C'


def test_verbose_answer_ignores_word_after_answer_is(matching_util):
    # The letter after "answer is" belongs to a word, not to an option.
    assert matching_util.can_infer_option('The answer is clearly B, not A.', dict(CHOICES)) is False
    assert matching_util.can_infer_option('The answer is definitely B rather than A', dict(CHOICES)) is False


def test_verbose_answer_ignores_lowercase_article(matching_util):
    # "a" is an article here; the option text identifies B.
    assert matching_util.can_infer('The answer is a dog.', dict(CHOICES)) == 'B'


def test_extract_characters_regex_keeps_option_letters(multiple_choice):
    assert multiple_choice.extract_characters_regex('B') == 'B'
    assert multiple_choice.extract_characters_regex('The answer is (C)') == 'C'
    assert multiple_choice.extract_characters_regex('b') == 'B'
    assert multiple_choice.extract_characters_regex('(d)') == 'D'


@pytest.mark.parametrize('prediction', ['', '   ', 'The answer is', 'The best answer is', '('])
def test_extract_characters_regex_rejects_empty_prediction(multiple_choice, prediction):
    # An answer with no option in it must be rejected, not scored as option A.
    assert multiple_choice.extract_characters_regex(prediction) == ''
