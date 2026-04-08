# tests/test_layer1.py
"""Integration tests for Layer 1 deterministic extraction."""
import pytest
from server.ontology.schema import Concept, Category, Dimension, Ontology
from server.extract.layer1 import extract_layer1


def _make_test_ontology():
    concepts = [
        Concept(id="product_affinity.leather.travel_bag", label="Travel Bag",
                aliases={"en": ["travel bag"], "fr": ["sac de voyage"]}, weight=0.8),
        Concept(id="material_craftsmanship.leather.monogram_canvas", label="Monogram Canvas",
                aliases={"en": ["monogram", "monogramme"], "fr": ["monogramme"]}, weight=0.5),
        Concept(id="occasion.life_events.anniversary", label="Anniversary",
                aliases={"en": ["anniversary"], "fr": ["anniversaire"]}, weight=0.7),
    ]
    cat1 = Category(id="product_affinity.leather", label="Leather", concepts=[concepts[0]])
    cat2 = Category(id="material_craftsmanship.leather", label="Leather Materials", concepts=[concepts[1]])
    cat3 = Category(id="occasion.life_events", label="Life Events", concepts=[concepts[2]])
    dim1 = Dimension(id="product_affinity", label="Product Affinity", categories=[cat1])
    dim2 = Dimension(id="material_craftsmanship", label="Material", categories=[cat2])
    dim3 = Dimension(id="occasion", label="Occasion", categories=[cat3])
    return Ontology(dimensions=[dim1, dim2, dim3])


def test_layer1_extracts_concepts_and_budget():
    ont = _make_test_ontology()
    text = "Cherche un sac de voyage pour anniversaire, budget 3000€"
    result = extract_layer1(text, ontology=ont, lang="fr")
    assert "concepts" in result
    assert "budgets" in result
    assert "negations" in result
    concept_ids = [c["concept_id"] for c in result["concepts"]]
    assert "product_affinity.leather.travel_bag" in concept_ids
    assert "occasion.life_events.anniversary" in concept_ids
    assert len(result["budgets"]) >= 1


def test_layer1_negation_creates_rejection():
    ont = _make_test_ontology()
    text = "Pas de monogramme, cherche un sac de voyage"
    result = extract_layer1(text, ontology=ont, lang="fr")
    concept_ids = [c["concept_id"] for c in result["concepts"]]
    # "sac de voyage" should be a positive match
    assert "product_affinity.leather.travel_bag" in concept_ids
    # "monogramme" should be in rejections, not in positive concepts
    assert "material_craftsmanship.leather.monogram_canvas" not in concept_ids
    rejection_ids = [r["concept_id"] for r in result["rejections"]]
    assert "material_craftsmanship.leather.monogram_canvas" in rejection_ids


def test_layer1_all_fields_have_tier_stated():
    ont = _make_test_ontology()
    text = "Un sac de voyage"
    result = extract_layer1(text, ontology=ont, lang="fr")
    for c in result["concepts"]:
        assert c["tier"] == "stated"
    for b in result["budgets"]:
        assert b["tier"] == "stated"
