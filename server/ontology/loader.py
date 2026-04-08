"""Load and save the hierarchical ontology from/to JSON."""
from __future__ import annotations
import json
from pathlib import Path

from server.ontology.schema import (
    Concept, Category, Dimension, Ontology,
    Relationship, RelationshipType,
)


def load_ontology(path: Path) -> Ontology:
    """Load ontology from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dimensions = []
    for dim_data in data.get("dimensions", []):
        categories = []
        for cat_data in dim_data.get("categories", []):
            concepts = []
            for con_data in cat_data.get("concepts", []):
                rels = []
                for rel_data in con_data.get("relationships", []):
                    rels.append(Relationship(
                        type=RelationshipType(rel_data["type"]),
                        target_id=rel_data["target_id"],
                        weight=rel_data.get("weight", 1.0),
                        metadata=rel_data.get("metadata", {}),
                    ))
                concepts.append(Concept(
                    id=con_data["id"],
                    label=con_data["label"],
                    aliases=con_data.get("aliases", {}),
                    weight=con_data.get("weight", 1.0),
                    relationships=rels,
                ))
            categories.append(Category(
                id=cat_data["id"],
                label=cat_data["label"],
                concepts=concepts,
            ))
        dimensions.append(Dimension(
            id=dim_data["id"],
            label=dim_data["label"],
            categories=categories,
        ))
    return Ontology(dimensions=dimensions)


def save_ontology(ontology: Ontology, path: Path) -> None:
    """Save ontology to a JSON file."""
    data: dict = {"dimensions": []}
    for dim in ontology.dimensions:
        dim_data: dict = {"id": dim.id, "label": dim.label, "categories": []}
        for cat in dim.categories:
            cat_data: dict = {"id": cat.id, "label": cat.label, "concepts": []}
            for con in cat.concepts:
                con_data: dict = {
                    "id": con.id,
                    "label": con.label,
                    "aliases": con.aliases,
                    "weight": con.weight,
                    "relationships": [
                        {
                            "type": rel.type.value,
                            "target_id": rel.target_id,
                            "weight": rel.weight,
                            **({"metadata": rel.metadata} if rel.metadata else {}),
                        }
                        for rel in con.relationships
                    ],
                }
                cat_data["concepts"].append(con_data)
            dim_data["categories"].append(cat_data)
        data["dimensions"].append(dim_data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
