# tests/test_ontology.py
"""Tests for the hierarchical ontology data model."""
import json
import tempfile
from pathlib import Path
import pytest
from server.ontology.schema import (
    Concept, Category, Dimension, Ontology, Relationship, RelationshipType
)
from server.ontology.relationships import RelationshipGraph
from server.ontology.loader import load_ontology, save_ontology


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


def _make_minimal_ontology_json():
    return {
        "dimensions": [
            {
                "id": "product_affinity",
                "label": "Product Affinity",
                "categories": [
                    {
                        "id": "product_affinity.watches",
                        "label": "Watches",
                        "concepts": [
                            {
                                "id": "product_affinity.watches.tourbillon_interest",
                                "label": "Tourbillon Interest",
                                "aliases": {"en": ["tourbillon"], "fr": ["tourbillon"]},
                                "weight": 0.8,
                                "relationships": [
                                    {"type": "implies", "target_id": "lifestyle.hobbies.watch_collector", "weight": 0.7}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }


def test_load_ontology_from_json():
    data = _make_minimal_ontology_json()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        ont = load_ontology(Path(f.name))
    assert len(ont.dimensions) == 1
    assert len(ont.all_concepts()) == 1
    c = ont.get_concept("product_affinity.watches.tourbillon_interest")
    assert c is not None
    assert c.weight == 0.8
    assert len(c.relationships) == 1
    assert c.relationships[0].type == RelationshipType.IMPLIES


def test_save_and_reload_ontology():
    data = _make_minimal_ontology_json()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        ont = load_ontology(Path(f.name))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
        save_ontology(ont, Path(f2.name))
        ont2 = load_ontology(Path(f2.name))

    assert len(ont2.all_concepts()) == len(ont.all_concepts())


def test_load_ontology_validates_ids():
    data = _make_minimal_ontology_json()
    data["dimensions"][0]["categories"][0]["concepts"][0]["id"] = ""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        with pytest.raises(ValueError):
            load_ontology(Path(f.name))


def test_relationship_graph_implies():
    c1 = Concept(
        id="a.b.c1", label="C1", aliases={"en": ["x"]}, weight=1.0,
        relationships=[Relationship(type=RelationshipType.IMPLIES, target_id="a.b.c2", weight=0.8)],
    )
    c2 = Concept(id="a.b.c2", label="C2", aliases={"en": ["y"]}, weight=1.0)
    cat = Category(id="a.b", label="B", concepts=[c1, c2])
    dim = Dimension(id="a", label="A", categories=[cat])
    ont = Ontology(dimensions=[dim])
    graph = RelationshipGraph(ont)

    implied = graph.get_implied("a.b.c1")
    assert "a.b.c2" in [r["target_id"] for r in implied]


def test_relationship_graph_conflicts():
    c1 = Concept(
        id="a.b.c1", label="C1", aliases={"en": ["x"]}, weight=1.0,
        relationships=[Relationship(type=RelationshipType.CONFLICTS, target_id="a.b.c2")],
    )
    c2 = Concept(id="a.b.c2", label="C2", aliases={"en": ["y"]}, weight=1.0)
    cat = Category(id="a.b", label="B", concepts=[c1, c2])
    dim = Dimension(id="a", label="A", categories=[cat])
    ont = Ontology(dimensions=[dim])
    graph = RelationshipGraph(ont)

    conflicts = graph.get_conflicts("a.b.c1")
    assert "a.b.c2" in conflicts


from server.shared.config import TAXONOMY_DIR


def test_seed_ontology_loads_and_has_14_dimensions():
    path = TAXONOMY_DIR / "ontology_v2.json"
    if not path.exists():
        pytest.skip("ontology_v2.json not yet generated")
    ont = load_ontology(path)
    assert len(ont.dimensions) == 14
    concepts = ont.all_concepts()
    assert len(concepts) >= 200  # minimum viable ontology


def test_seed_ontology_no_empty_aliases():
    path = TAXONOMY_DIR / "ontology_v2.json"
    if not path.exists():
        pytest.skip("ontology_v2.json not yet generated")
    ont = load_ontology(path)
    for c in ont.all_concepts():
        assert len(c.all_aliases()) > 0, f"Concept {c.id} has no aliases"


def test_seed_ontology_no_duplicate_ids():
    path = TAXONOMY_DIR / "ontology_v2.json"
    if not path.exists():
        pytest.skip("ontology_v2.json not yet generated")
    ont = load_ontology(path)
    ids = [c.id for c in ont.all_concepts()]
    assert len(ids) == len(set(ids)), "Duplicate concept IDs found"
