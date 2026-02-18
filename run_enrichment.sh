#!/bin/bash
# Overnight Ollama Enrichment Runner
cd /Users/ian/BDD2-LVMH
source .venv/bin/activate

echo "Starting Ollama Vocabulary Enrichment..."
echo "Model: qwen2.5:3b"
echo "Phase: all (5 phases)"
echo "Time: $(date)"
echo "Logs: tail -f ollama_enrichment.log"
echo "---"

python -m server.vocabulary.ollama_enrichment --model qwen2.5:3b --phase all 2>&1 | tee ollama_enrichment.log

echo ""
echo "=== ENRICHMENT COMPLETE ==="
echo "Time: $(date)"
