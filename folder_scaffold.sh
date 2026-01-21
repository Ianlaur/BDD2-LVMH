#!/usr/bin/env bash
set -e

mkdir -p \
  data/raw data/processed data/outputs \
  docs taxonomy activations demo/figures \
  schemas notebooks \
  src/ingest src/extract src/lexicon src/embeddings src/profiling src/actions src/utils

touch docs/pipeline_documentation.md docs/data_dictionary.md docs/evaluation_report.md
touch activations/playbooks.yml
touch schemas/extraction_output.schema.json schemas/playbook.schema.json schemas/taxonomy.schema.json

touch src/__init__.py
touch src/ingest/__init__.py src/extract/__init__.py src/lexicon/__init__.py \
      src/embeddings/__init__.py src/profiling/__init__.py src/actions/__init__.py src/utils/__init__.py

# Only create stub files if they do NOT exist (prevents overwriting your work)
if [ ! -f src/run_all.py ]; then
cat > src/run_all.py <<'PY'
"""
Run the full deterministic (non-LLM) pipeline.
This is a stub scaffold; implement each stage module to make it functional.
"""
from src.ingest.run_ingest import main as ingest_main
from src.extract.run_candidates import main as candidates_main
from src.lexicon.build_lexicon import main as lexicon_main
from src.extract.detect_concepts import main as concepts_main
from src.embeddings.build_vectors import main as vectors_main
from src.profiling.segment_clients import main as profiles_main
from src.actions.recommend_actions import main as actions_main

def main():
    ingest_main()
    candidates_main()
    lexicon_main()
    concepts_main()
    vectors_main()
    profiles_main()
    actions_main()

if __name__ == "__main__":
    main()
PY
fi

if [ ! -f src/ingest/run_ingest.py ]; then
cat > src/ingest/run_ingest.py <<'PY'
def main():
    # TODO: load CSV from data/raw, validate schema, clean, write data/processed/notes_clean.parquet
    print("[ingest] TODO")
PY
fi

if [ ! -f src/extract/run_candidates.py ]; then
cat > src/extract/run_candidates.py <<'PY'
def main():
    # TODO: extract keyphrases via TF-IDF/YAKE/RAKE and write data/processed/candidates.csv
    print("[candidates] TODO")
PY
fi

if [ ! -f src/lexicon/build_lexicon.py ]; then
cat > src/lexicon/build_lexicon.py <<'PY'
def main():
    # TODO: build lexicon_v1.csv + taxonomy_v1.json from candidates.csv
    print("[lexicon] TODO")
PY
fi

if [ ! -f src/extract/detect_concepts.py ]; then
cat > src/extract/detect_concepts.py <<'PY'
def main():
    # TODO: detect concepts in notes using lexicon rules; output extracted_concepts + flat tables
    print("[concepts] TODO")
PY
fi

if [ ! -f src/embeddings/build_vectors.py ]; then
cat > src/embeddings/build_vectors.py <<'PY'
def main():
    # TODO: build note/client/concept vectors (TF-IDF and/or fastText), persist to data/outputs
    print("[vectors] TODO")
PY
fi

if [ ! -f src/embeddings/projection_3d.py ]; then
cat > src/embeddings/projection_3d.py <<'PY'
def main():
    # TODO: project vectors to 3D (PCA/UMAP) and write demo/embedding_space_3d.html
    print("[viz3d] TODO")
PY
fi

if [ ! -f src/profiling/segment_clients.py ]; then
cat > src/profiling/segment_clients.py <<'PY'
def main():
    # TODO: cluster vectors and label profile types; output data/outputs/client_profiles.csv
    print("[profiles] TODO")
PY
fi

if [ ! -f src/actions/recommend_actions.py ]; then
cat > src/actions/recommend_actions.py <<'PY'
def main():
    # TODO: apply activations/playbooks.yml to profiles/concepts; output recommended_actions.csv
    print("[actions] TODO")
PY
fi

# Create requirements.txt only if missing
if [ ! -f requirements.txt ]; then
cat > requirements.txt <<'REQ'
pandas==2.2.2
numpy==2.0.1
scikit-learn==1.5.1
yake==0.4.8
rake-nltk==1.0.6
umap-learn==0.5.6
plotly==5.23.0
REQ
fi

touch data/raw/.gitkeep

echo "Scaffold created."
echo "Next:"
echo "  1) Put your CSV in data/raw/"
echo "  2) Implement the TODOs in src/*"
echo "  3) make build"
echo "  4) make run"