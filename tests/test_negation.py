# tests/test_negation.py
"""Tests for multilingual negation detection."""
import pytest
from server.extract.negation import detect_negation_spans


def test_french_pas_de():
    text = "Elle ne veut pas de monogramme sur le sac"
    spans = detect_negation_spans(text, lang="fr")
    assert len(spans) >= 1
    # The negation span should cover "monogramme"
    neg_text = text[spans[0]["start"]:spans[0]["end"]]
    assert "monogramme" in neg_text


def test_french_sans():
    text = "Un sac sans logo visible"
    spans = detect_negation_spans(text, lang="fr")
    assert len(spans) >= 1
    neg_text = text[spans[0]["start"]:spans[0]["end"]]
    assert "logo" in neg_text


def test_english_no():
    text = "No monogram please, something discreet"
    spans = detect_negation_spans(text, lang="en")
    assert len(spans) >= 1
    neg_text = text[spans[0]["start"]:spans[0]["end"]]
    assert "monogram" in neg_text


def test_english_not():
    text = "She does not like bright colors"
    spans = detect_negation_spans(text, lang="en")
    assert len(spans) >= 1
    neg_text = text[spans[0]["start"]:spans[0]["end"]]
    assert "bright colors" in neg_text


def test_negation_scope_limited():
    """Negation should not extend beyond 5 tokens."""
    text = "pas de logo mais elle adore le cuir et les montres vintage"
    spans = detect_negation_spans(text, lang="fr")
    assert len(spans) == 1
    neg_text = text[spans[0]["start"]:spans[0]["end"]]
    # "cuir" and "montres" should NOT be in the negation span
    assert "cuir" not in neg_text
    assert "montres" not in neg_text


def test_no_negation_returns_empty():
    text = "Elle cherche un beau sac en cuir"
    spans = detect_negation_spans(text, lang="fr")
    assert spans == []


def test_multiple_negations():
    text = "Pas de monogramme, pas de couleurs vives"
    spans = detect_negation_spans(text, lang="fr")
    assert len(spans) == 2


def test_german_negation():
    text = "Kein Logo bitte, etwas Dezentes"
    spans = detect_negation_spans(text, lang="de")
    assert len(spans) >= 1


def test_spanish_negation():
    text = "Sin logo visible por favor"
    spans = detect_negation_spans(text, lang="es")
    assert len(spans) >= 1


def test_italian_negation():
    text = "Non vuole il monogramma"
    spans = detect_negation_spans(text, lang="it")
    assert len(spans) >= 1
