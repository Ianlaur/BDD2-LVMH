"""Data classes for the 14-dimension hierarchical ontology."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class RelationshipType(Enum):
    IMPLIES = "implies"
    CONFLICTS = "conflicts"
    AMPLIFIES = "amplifies"
    CO_OCCURS = "co_occurs"
    UPGRADES_TO = "upgrades_to"
    SUBSTITUTES = "substitutes"
    SEASONAL = "seasonal"


@dataclass
class Relationship:
    type: RelationshipType
    target_id: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class Concept:
    id: str                                     # "dimension.category.concept_name"
    label: str                                  # Human-readable label
    aliases: dict[str, list[str]]               # {"en": ["watch"], "fr": ["montre"]}
    weight: float                               # Differentiating power (0-1, higher = rarer/more valuable)
    relationships: list[Relationship] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            raise ValueError("Concept id cannot be empty")

    def all_aliases(self) -> list[str]:
        """Return flat deduplicated list of all aliases across all languages."""
        seen = set()
        result = []
        for lang_aliases in self.aliases.values():
            for alias in lang_aliases:
                lower = alias.lower()
                if lower not in seen:
                    seen.add(lower)
                    result.append(alias)
        return result


@dataclass
class Category:
    id: str                                     # "dimension.category_name"
    label: str
    concepts: list[Concept] = field(default_factory=list)


@dataclass
class Dimension:
    id: str                                     # "dimension_name"
    label: str
    categories: list[Category] = field(default_factory=list)


@dataclass
class Ontology:
    dimensions: list[Dimension] = field(default_factory=list)
    _concept_index: dict[str, Concept] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._rebuild_index()

    def _rebuild_index(self):
        self._concept_index = {}
        for dim in self.dimensions:
            for cat in dim.categories:
                for concept in cat.concepts:
                    self._concept_index[concept.id] = concept

    def all_concepts(self) -> list[Concept]:
        return list(self._concept_index.values())

    def get_concept(self, concept_id: str) -> Concept | None:
        return self._concept_index.get(concept_id)
