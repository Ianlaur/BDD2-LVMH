"""Queryable relationship graph built from ontology concept relationships."""
from __future__ import annotations
from collections import defaultdict

from server.ontology.schema import Ontology, RelationshipType


class RelationshipGraph:
    """Provides fast lookup of concept relationships."""

    def __init__(self, ontology: Ontology):
        self._implies: dict[str, list[dict]] = defaultdict(list)
        self._conflicts: dict[str, set[str]] = defaultdict(set)
        self._amplifies: dict[str, list[dict]] = defaultdict(list)
        self._co_occurs: dict[str, set[str]] = defaultdict(set)

        for concept in ontology.all_concepts():
            for rel in concept.relationships:
                if rel.type == RelationshipType.IMPLIES:
                    self._implies[concept.id].append({
                        "target_id": rel.target_id, "weight": rel.weight,
                    })
                elif rel.type == RelationshipType.CONFLICTS:
                    self._conflicts[concept.id].add(rel.target_id)
                    self._conflicts[rel.target_id].add(concept.id)
                elif rel.type == RelationshipType.AMPLIFIES:
                    self._amplifies[concept.id].append({
                        "target_id": rel.target_id, "weight": rel.weight,
                    })
                elif rel.type == RelationshipType.CO_OCCURS:
                    self._co_occurs[concept.id].add(rel.target_id)
                    self._co_occurs[rel.target_id].add(concept.id)

    def get_implied(self, concept_id: str) -> list[dict]:
        return self._implies.get(concept_id, [])

    def get_conflicts(self, concept_id: str) -> set[str]:
        return self._conflicts.get(concept_id, set())

    def get_amplifies(self, concept_id: str) -> list[dict]:
        return self._amplifies.get(concept_id, [])

    def get_co_occurs(self, concept_id: str) -> set[str]:
        return self._co_occurs.get(concept_id, set())
