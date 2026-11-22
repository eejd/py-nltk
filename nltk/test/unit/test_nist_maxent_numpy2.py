"""Regression tests: NumPy 2 removed ``numpy.NINF``; NIST must not divide by zero."""

import pytest

from nltk.classify.maxent import train_maxent_classifier_with_gis as gis
from nltk.classify.maxent import train_maxent_classifier_with_iis as iis
from nltk.translate.nist_score import sentence_nist

TRAIN = [
    ({"x": 1}, "a"),
    ({"x": 2}, "b"),
    ({"x": 1, "y": 1}, "a"),
    ({"y": 2}, "b"),
]


@pytest.mark.parametrize("trainer", [gis, iis])
def test_maxent_trains_with_unattested_features(trainer):
    # Unattested features get -inf weights; numpy.NINF raised AttributeError on NumPy 2.
    clf = trainer(TRAIN, max_iter=5, trace=0)
    assert clf.classify({"x": 1}) in {"a", "b"}


def test_nist_short_hypothesis_does_not_divide_by_zero():
    ref = "It is a guide to action that ensures that the military".split()
    score = sentence_nist([ref], ["a", "b"], n=5)
    assert isinstance(score, float)
    assert score >= 0


def test_nist_order_longer_than_hypothesis_scores_zero_precision():
    assert sentence_nist([["a", "b", "c"]], ["a"], n=3) >= 0
