# tests/test_ontology.py
"""Tests for the hierarchical ontology data model."""
import pytest
from server.ontology.schema import (
    Concept, Category, Dimension, Ontology, RelationshipType
)


def test_concept_creation():
    c = Concept(
        id="product_affinity.watches.tourbillon_interest",
        label="Tourbillon Interest",
        aliases={"en": ["tourbillon"], "fr": ["tourbillon", "mouvement tourbillon"]},
        weight=0.8,
    )
    assert c.id == "product_affinity.watches.tourbillon_interest"
    assert "fr" in c.aliases
    assert c.weight == 0.8


def test_concept_all_aliases_flat():
    """all_aliases() returns a flat deduplicated list across all languages."""
    c = Concept(
        id="test.cat.concept",
        label="Test",
        aliases={"en": ["watch", "timepiece"], "fr": ["montre", "watch"]},
        weight=1.0,
    )
    flat = c.all_aliases()
    assert "watch" in flat
    assert "montre" in flat
    assert "timepiece" in flat
    # deduplicated
    assert len(flat) == len(set(flat))


def test_concept_rejects_empty_id():
    with pytest.raises(ValueError):
        Concept(id="", label="X", aliases={"en": ["x"]}, weight=1.0)


def test_category_contains_concepts():
    c1 = Concept(id="d.cat.c1", label="C1", aliases={"en": ["a"]}, weight=1.0)
    c2 = Concept(id="d.cat.c2", label="C2", aliases={"en": ["b"]}, weight=1.0)
    cat = Category(id="d.cat", label="Cat", concepts=[c1, c2])
    assert len(cat.concepts) == 2


def test_dimension_contains_categories():
    c = Concept(id="d.cat.c1", label="C1", aliases={"en": ["a"]}, weight=1.0)
    cat = Category(id="d.cat", label="Cat", concepts=[c])
    dim = Dimension(id="d", label="D", categories=[cat])
    assert dim.categories[0].id == "d.cat"


def test_ontology_get_all_concepts():
    c1 = Concept(id="d.cat.c1", label="C1", aliases={"en": ["alpha"]}, weight=1.0)
    c2 = Concept(id="d.cat.c2", label="C2", aliases={"en": ["beta"]}, weight=1.0)
    cat = Category(id="d.cat", label="Cat", concepts=[c1, c2])
    dim = Dimension(id="d", label="D", categories=[cat])
    ont = Ontology(dimensions=[dim])
    assert len(ont.all_concepts()) == 2


def test_ontology_get_concept_by_id():
    c1 = Concept(id="d.cat.c1", label="C1", aliases={"en": ["alpha"]}, weight=1.0)
    cat = Category(id="d.cat", label="Cat", concepts=[c1])
    dim = Dimension(id="d", label="D", categories=[cat])
    ont = Ontology(dimensions=[dim])
    assert ont.get_concept("d.cat.c1") is c1
    assert ont.get_concept("nonexistent") is None


def test_relationship_types():
    assert RelationshipType.IMPLIES.value == "implies"
    assert RelationshipType.CONFLICTS.value == "conflicts"
    assert RelationshipType.AMPLIFIES.value == "amplifies"
