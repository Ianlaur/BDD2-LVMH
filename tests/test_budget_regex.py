# tests/test_budget_regex.py
"""Tests for budget/amount regex extraction."""
import pytest
from server.extract.budget_regex import extract_budgets


def test_euro_amount():
    matches = extract_budgets("budget autour de 3000€")
    assert len(matches) == 1
    assert matches[0]["amount_text"] == "3000€"


def test_range_with_k():
    matches = extract_budgets("budget 3-4k")
    assert len(matches) == 1
    assert "3-4k" in matches[0]["amount_text"]


def test_price_around():
    matches = extract_budgets("price around 40K+")
    assert len(matches) == 1


def test_spanish_budget():
    matches = extract_budgets("presupuesto 25-30k euros")
    assert len(matches) == 1


def test_no_budget():
    matches = extract_budgets("Elle cherche un sac classique")
    assert matches == []


def test_multiple_amounts():
    matches = extract_budgets("budget 3000€ à 5000€")
    assert len(matches) >= 1


def test_budget_returns_evidence_span():
    text = "Son budget est autour de 5000€ pour un cadeau"
    matches = extract_budgets(text)
    assert len(matches) == 1
    assert "start" in matches[0]
    assert "end" in matches[0]
    assert text[matches[0]["start"]:matches[0]["end"]]  # non-empty span
