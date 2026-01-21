FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app \
    HF_HOME=/app/models/hf \
    TRANSFORMERS_CACHE=/app/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers

WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ${APP_HOME}/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download NLTK assets used by rake-nltk (stopwords, punkt)
RUN python - <<'PY'
import nltk
nltk.download('stopwords')
nltk.download('punkt')
PY

COPY . ${APP_HOME}

# Create expected runtime directories (also fine if mounted)
RUN mkdir -p data/raw data/processed data/outputs demo docs taxonomy activations models/hf models/sentence_transformers

CMD ["make", "pipeline"]