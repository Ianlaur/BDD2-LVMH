"""
Main pipeline orchestrator: Run all stages in sequence.

Usage:
    python -m src.run_all
    
Or via make:
    make dev          (local venv)
    make run          (Docker)
"""
import sys
import time
from pathlib import Path

from src.utils import log_stage, ensure_directories, set_all_seeds


def run_pipeline():
    """Run the complete pipeline."""
    start_time = time.time()
    
    print("=" * 60)
    print("LVMH Client Intelligence Pipeline")
    print("Hybrid (Rule-Based + LLM) Multilingual Processing")
    print("=" * 60)
    print()
    
    # Ensure all directories exist
    ensure_directories()
    set_all_seeds()
    
    # Track stage timings
    timings = {}
    
    try:
        # Stage 1: Ingest
        print("\n" + "=" * 40)
        print("STAGE 1: INGEST")
        print("=" * 40)
        stage_start = time.time()
        from src.ingest.run_ingest import run_ingest
        run_ingest()
        timings["ingest"] = time.time() - stage_start
        
        # Stage 2: Candidate Extraction
        print("\n" + "=" * 40)
        print("STAGE 2: CANDIDATE EXTRACTION")
        print("=" * 40)
        stage_start = time.time()
        from src.extract.run_candidates import run_candidates
        run_candidates()
        timings["candidates"] = time.time() - stage_start
        
        # Stage 3: Lexicon Building
        print("\n" + "=" * 40)
        print("STAGE 3: LEXICON BUILDING")
        print("=" * 40)
        stage_start = time.time()
        from src.lexicon.build_lexicon import build_lexicon
        build_lexicon()
        timings["lexicon"] = time.time() - stage_start
        
        # Stage 4: Concept Detection
        print("\n" + "=" * 40)
        print("STAGE 4: CONCEPT DETECTION")
        print("=" * 40)
        stage_start = time.time()
        from src.extract.detect_concepts import detect_concepts
        detect_concepts()
        timings["concepts"] = time.time() - stage_start
        
        # Stage 5: Vector Building
        print("\n" + "=" * 40)
        print("STAGE 5: VECTOR BUILDING")
        print("=" * 40)
        stage_start = time.time()
        from src.embeddings.build_vectors import build_vectors
        build_vectors()
        timings["vectors"] = time.time() - stage_start
        
        # Stage 6: Client Segmentation
        print("\n" + "=" * 40)
        print("STAGE 6: CLIENT SEGMENTATION")
        print("=" * 40)
        stage_start = time.time()
        from src.profiling.segment_clients import segment_clients
        segment_clients()
        timings["profiles"] = time.time() - stage_start
        
        # Stage 7: Action Recommendation
        print("\n" + "=" * 40)
        print("STAGE 7: ACTION RECOMMENDATION")
        print("=" * 40)
        stage_start = time.time()
        from src.actions.recommend_actions import recommend_actions
        recommend_actions()
        timings["actions"] = time.time() - stage_start
        
        # Stage 8: 3D Projection (Optional)
        print("\n" + "=" * 40)
        print("STAGE 8: 3D PROJECTION (Optional)")
        print("=" * 40)
        stage_start = time.time()
        try:
            from src.embeddings.projection_3d import project_3d
            project_3d()
            timings["projection"] = time.time() - stage_start
        except Exception as e:
            log_stage("projection", f"Skipped: {e}")
            timings["projection"] = 0
        
    except Exception as e:
        log_stage("pipeline", f"PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nTotal time: {total_time:.1f}s")
    print("\nStage timings:")
    for stage, duration in timings.items():
        print(f"  {stage}: {duration:.1f}s")
    
    # List outputs
    print("\nOutputs generated:")
    from src.config import DATA_PROCESSED, DATA_OUTPUTS, TAXONOMY_DIR, DEMO_DIR
    
    outputs = [
        DATA_PROCESSED / "notes_clean.parquet",
        DATA_PROCESSED / "candidates.csv",
        TAXONOMY_DIR / "lexicon_v1.csv",
        TAXONOMY_DIR / "taxonomy_v1.json",
        DATA_OUTPUTS / "note_concepts.csv",
        DATA_OUTPUTS / "note_vectors.parquet",
        DATA_OUTPUTS / "client_profiles.csv",
        DATA_OUTPUTS / "recommended_actions.csv",
        DEMO_DIR / "embedding_space_3d.html"
    ]
    
    for output in outputs:
        if output.exists():
            size = output.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            print(f"  ✓ {output.relative_to(output.parent.parent)} ({size_str})")
        else:
            print(f"  ✗ {output.relative_to(output.parent.parent)} (not created)")
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)


def main():
    """CLI entry point."""
    run_pipeline()


if __name__ == "__main__":
    main()
