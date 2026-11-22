"""Regression tests: NumPy 2 removed ``numpy.NINF``; NIST must not divide by zero."""

import numpy

from nltk.classify.maxent import GISEncoding
from nltk.classify.maxent import train_maxent_classifier_with_gis as gis
from nltk.translate.nist_score import sentence_nist

TRAIN = [
    ({"x": 1}, "a"),
    ({"x": 2}, "b"),
    ({"x": 1, "y": 1}, "a"),
    ({"y": 2}, "b"),
]


def test_gis_unattested_feature_without_numpy_NINF(monkeypatch):
    # The mapping includes ('z', 1, 'a'), which never occurs in TRAIN, so GIS
    # must assign it -inf. Removing numpy.NINF (NumPy 2) must not break this.
    monkeypatch.delattr(numpy, "NINF", raising=False)
    mapping = {
        ("x", 1, "a"): 0,
        ("x", 1, "b"): 1,
        ("y", 1, "a"): 2,
        ("y", 2, "b"): 3,
        ("z", 1, "a"): 4,
    }
    encoding = GISEncoding(["a", "b"], mapping, C=10)
    clf = gis(TRAIN, encoding=encoding, max_iter=5, trace=0)
    assert clf.classify({"x": 1}) in {"a", "b"}


def test_nist_short_hypothesis_does_not_divide_by_zero():
    ref = "It is a guide to action that ensures that the military".split()
    score = sentence_nist([ref], ["a", "b"], n=5)
    assert isinstance(score, float)
    assert score >= 0


def test_nist_order_longer_than_hypothesis_scores_zero_precision():
    assert sentence_nist([["a", "b", "c"]], ["a"], n=3) >= 0
