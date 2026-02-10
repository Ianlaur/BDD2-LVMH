"""
FastAPI server for LVMH Dashboard - Remote access to data and pipeline execution.

Usage:
    python -m server.api_server
    
    Or with custom host/port:
    python -m server.api_server --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json
import argparse
import uvicorn
import os
from typing import Optional
import logging
import shutil
from datetime import datetime

from server.shared.config import BASE_DIR, DATA_OUTPUTS, TAXONOMY_DIR, DATA_INPUT, DATA_PROCESSED
from server.run_all import run_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LVMH Voice-to-Tag API",
    description="API for accessing LVMH client intelligence data and running the pipeline",
    version="1.0.0"
)

# Configure CORS - Allow dashboard to connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact dashboard URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline status tracking
pipeline_status = {
    "running": False,
    "last_run": None,
    "last_error": None
}


@app.get("/")
async def root():
    """API health check and info."""
    return {
        "service": "LVMH Voice-to-Tag API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "data": "/api/data",
            "pipeline_status": "/api/pipeline/status",
            "run_pipeline": "/api/pipeline/run",
            "outputs": "/api/outputs/{filename}"
        }
    }


@app.get("/api/data")
@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Get the main dashboard data (data.json)."""
    data_path = BASE_DIR / "dashboard" / "src" / "data.json"
    
    if not data_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="Dashboard data not found. Run the pipeline first."
        )
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clients")
async def get_clients():
    """Get all client profiles."""
    # Try to get client profiles with predictions first
    predictions_path = DATA_OUTPUTS / "client_profiles_with_predictions.csv"
    profiles_path = DATA_OUTPUTS / "client_profiles.json"
    
    if predictions_path.exists():
        import pandas as pd
        import numpy as np
        df = pd.read_csv(predictions_path)
        # Replace NaN with None for JSON serialization
        df = df.replace({np.nan: None})
        return df.to_dict('records')
    elif profiles_path.exists():
        with open(profiles_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise HTTPException(
            status_code=404, 
            detail="Client profiles not found. Run the pipeline first."
        )


@app.get("/api/clients/{client_id}")
async def get_client(client_id: str):
    """Get a specific client's profile."""
    clients = await get_clients()
    for client in clients:
        if str(client.get('client_id')) == client_id or str(client.get('id')) == client_id:
            return client
    raise HTTPException(status_code=404, detail=f"Client {client_id} not found")


@app.get("/api/predictions")
async def get_predictions():
    """Get all ML predictions."""
    predictions_path = DATA_OUTPUTS / "client_profiles_with_predictions.csv"
    
    if not predictions_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="ML predictions not found. Run the pipeline with ML predictions enabled."
        )
    
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(predictions_path)
        # Filter to only prediction columns
        pred_cols = ['client_id', 'purchase_probability', 'churn_risk', 
                     'predicted_clv', 'value_segment']
        available_cols = [col for col in pred_cols if col in df.columns]
        if 'client_id' not in df.columns and 'id' in df.columns:
            available_cols = ['id'] + [col for col in pred_cols[1:] if col in df.columns]
        
        # Replace NaN with None for JSON serialization
        df_filtered = df[available_cols].replace({np.nan: None})
        return df_filtered.to_dict('records')
    except Exception as e:
        logger.error(f"Error reading predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/{client_id}")
async def get_client_predictions(client_id: str):
    """Get ML predictions for a specific client."""
    predictions = await get_predictions()
    for pred in predictions:
        if str(pred.get('client_id')) == client_id or str(pred.get('id')) == client_id:
            return pred
    raise HTTPException(status_code=404, detail=f"Predictions for client {client_id} not found")


@app.get("/api/outputs/{filename}")
async def get_output_file(filename: str):
    """Get a specific output file (CSV, JSON, etc.)."""
    file_path = DATA_OUTPUTS / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get the current pipeline status."""
    return pipeline_status


@app.post("/api/pipeline/run")
async def run_pipeline_endpoint(
    background_tasks: BackgroundTasks,
    csv_path: Optional[str] = None,
    text_column: Optional[str] = None,
    id_column: Optional[str] = None,
    analyze_only: bool = False
):
    """
    Trigger the pipeline to run in the background.
    
    Parameters:
    - csv_path: Optional path to custom CSV file
    - text_column: Optional name of text column
    - id_column: Optional name of ID column
    - analyze_only: If true, only analyze the CSV structure
    """
    if pipeline_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running"
        )
    
    pipeline_status["running"] = True
    pipeline_status["last_error"] = None
    
    def run_pipeline_task():
        """Background task to run the pipeline."""
        try:
            logger.info("Starting pipeline execution...")
            run_pipeline(csv_path, text_column, id_column, analyze_only)
            pipeline_status["running"] = False
            pipeline_status["last_run"] = "success"
            logger.info("Pipeline completed successfully")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_status["running"] = False
            pipeline_status["last_error"] = str(e)
            pipeline_status["last_run"] = "error"
    
    background_tasks.add_task(run_pipeline_task)
    
    return {
        "status": "started",
        "message": "Pipeline execution started in background"
    }


@app.post("/api/upload-csv")
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    run_pipeline_after: bool = True
):
    """
    Upload a CSV file and optionally run the pipeline.
    
    Parameters:
    - file: CSV file to upload
    - run_pipeline_after: If true, automatically run pipeline after upload
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted"
        )
    
    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(file.filename).stem
    new_filename = f"{original_name}_{timestamp}.csv"
    file_path = DATA_INPUT / new_filename
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded file saved to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Clean up older uploaded CSVs to prevent accumulation
    # Keep only the file we just saved
    try:
        for old_csv in DATA_INPUT.glob("*.csv"):
            if old_csv != file_path:
                old_csv.unlink()
                logger.info(f"Cleaned up old upload: {old_csv.name}")
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")
    
    # Optionally run pipeline
    if run_pipeline_after:
        if pipeline_status["running"]:
            return {
                "status": "uploaded",
                "filename": new_filename,
                "path": str(file_path),
                "message": "File uploaded but pipeline is already running. Please try running it manually later.",
                "pipeline_status": "busy"
            }
        
        # Run pipeline with the uploaded file
        pipeline_status["running"] = True
        pipeline_status["last_error"] = None
        
        def run_pipeline_task():
            """Background task to run the pipeline."""
            try:
                logger.info(f"Running pipeline with uploaded file: {file_path}")
                run_pipeline(csv_path=str(file_path))
                pipeline_status["running"] = False
                pipeline_status["last_run"] = "success"
                logger.info("Pipeline completed successfully")
            except Exception as e:
                logger.error(f"Pipeline failed: {e}")
                pipeline_status["running"] = False
                pipeline_status["last_error"] = str(e)
                pipeline_status["last_run"] = "error"
        
        background_tasks.add_task(run_pipeline_task)
        
        return {
            "status": "uploaded_and_running",
            "filename": new_filename,
            "path": str(file_path),
            "message": "File uploaded and pipeline started",
            "pipeline_status": "running"
        }
    else:
        return {
            "status": "uploaded",
            "filename": new_filename,
            "path": str(file_path),
            "message": "File uploaded successfully. Run pipeline manually when ready.",
            "pipeline_status": "idle"
        }


@app.get("/api/lexicon")
async def get_lexicon():
    """Get the current lexicon/vocabulary."""
    lexicon_json = TAXONOMY_DIR / "lexicon_v1.json"
    lexicon_csv = TAXONOMY_DIR / "lexicon_v1.csv"
    
    if lexicon_json.exists():
        with open(lexicon_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif lexicon_csv.exists():
        return FileResponse(
            path=lexicon_csv,
            filename="lexicon_v1.csv",
            media_type="text/csv"
        )
    else:
        raise HTTPException(status_code=404, detail="Lexicon not found")


@app.get("/api/knowledge-graph")
async def get_knowledge_graph():
    """Get the knowledge graph in Cytoscape format."""
    kg_path = DATA_OUTPUTS / "knowledge_graph_cytoscape.json"
    
    if not kg_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Knowledge graph not found. Run the pipeline first."
        )
    
    with open(kg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ===================================================================
# RGPD / GDPR Compliance Endpoints
# ===================================================================

@app.get("/api/rgpd/audit")
async def rgpd_audit():
    """
    GDPR Compliance Audit — scan processed data for PII / Article 9 violations.
    Returns compliance rate, violation counts and detailed findings.
    """
    from server.privacy.compliance import ComplianceAuditor
    import pandas as pd

    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    if not notes_path.exists():
        raise HTTPException(status_code=404, detail="No processed data found. Run the pipeline first.")

    df = pd.read_parquet(notes_path)
    auditor = ComplianceAuditor()
    results = auditor.audit_dataset(df, text_column="text")

    # Strip the heavy per-record list for the API response
    summary = {k: v for k, v in results.items() if k != "detailed_results"}
    summary["sample_violations"] = [
        r for r in results["detailed_results"] if r["has_violations"]
    ][:10]
    return summary


@app.delete("/api/rgpd/erase/{client_id}")
async def rgpd_erase_client(client_id: str):
    """
    GDPR Article 17 — Right to Erasure (droit à l'effacement).
    Deletes all data associated with a specific client_id from:
      - notes_clean.parquet
      - note_concepts.csv
      - client_profiles.csv / client_profiles_with_predictions.csv
      - recommended_actions.csv
      - clustering_results.json
      - data.json (dashboard)
    Returns a summary of what was deleted.
    """
    import pandas as pd

    erased: dict = {}

    # --- 1. notes_clean.parquet ---
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    if notes_path.exists():
        df = pd.read_parquet(notes_path)
        before = len(df)
        df = df[df["client_id"].astype(str) != str(client_id)]
        after = len(df)
        df.to_parquet(notes_path, index=False)
        erased["notes_clean.parquet"] = before - after

    # --- 2. note_concepts.csv ---
    nc_path = DATA_PROCESSED / "note_concepts.csv"
    if nc_path.exists():
        df = pd.read_csv(nc_path)
        before = len(df)
        id_col = "client_id" if "client_id" in df.columns else "note_id"
        df = df[df[id_col].astype(str) != str(client_id)]
        after = len(df)
        df.to_csv(nc_path, index=False)
        erased["note_concepts.csv"] = before - after

    # --- 3. client profiles ---
    for fname in ["client_profiles.csv", "client_profiles_with_predictions.csv"]:
        fpath = DATA_OUTPUTS / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            before = len(df)
            id_col = "client_id" if "client_id" in df.columns else "id"
            df = df[df[id_col].astype(str) != str(client_id)]
            after = len(df)
            df.to_csv(fpath, index=False)
            erased[fname] = before - after

    # --- 4. recommended_actions.csv ---
    ra_path = DATA_OUTPUTS / "recommended_actions.csv"
    if ra_path.exists():
        df = pd.read_csv(ra_path)
        before = len(df)
        id_col = "client_id" if "client_id" in df.columns else "id"
        df = df[df[id_col].astype(str) != str(client_id)]
        after = len(df)
        df.to_csv(ra_path, index=False)
        erased["recommended_actions.csv"] = before - after

    # --- 5. clustering_results.json ---
    cr_path = DATA_OUTPUTS / "clustering_results.json"
    if cr_path.exists():
        try:
            with open(cr_path, "r", encoding="utf-8") as f:
                cr = json.load(f)
            if isinstance(cr, dict) and "clients" in cr:
                before = len(cr["clients"])
                cr["clients"] = [
                    c for c in cr["clients"]
                    if str(c.get("client_id", c.get("id", ""))) != str(client_id)
                ]
                erased["clustering_results.json"] = before - len(cr["clients"])
                with open(cr_path, "w", encoding="utf-8") as f:
                    json.dump(cr, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # --- 6. dashboard data.json ---
    data_json = BASE_DIR / "dashboard" / "src" / "data.json"
    if data_json.exists():
        try:
            with open(data_json, "r", encoding="utf-8") as f:
                dj = json.load(f)
            if isinstance(dj, dict) and "clients" in dj:
                before = len(dj["clients"])
                dj["clients"] = [
                    c for c in dj["clients"]
                    if str(c.get("id", c.get("client_id", ""))) != str(client_id)
                ]
                erased["data.json"] = before - len(dj["clients"])
                with open(data_json, "w", encoding="utf-8") as f:
                    json.dump(dj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    total = sum(erased.values())
    if total == 0:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found in any data file.")

    logger.info(f"RGPD ERASURE: client_id={client_id} — {total} records deleted across {len(erased)} files")

    return {
        "status": "erased",
        "client_id": client_id,
        "total_records_deleted": total,
        "details": erased,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/rgpd/export/{client_id}")
async def rgpd_export_client(client_id: str):
    """
    GDPR Article 20 — Right to Data Portability (droit à la portabilité).
    Returns ALL data held about a specific client_id in a single JSON payload.
    """
    import pandas as pd

    export: dict = {"client_id": client_id}

    # notes_clean
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    if notes_path.exists():
        df = pd.read_parquet(notes_path)
        rows = df[df["client_id"].astype(str) == str(client_id)]
        export["notes"] = rows.to_dict("records")

    # note_concepts
    nc_path = DATA_PROCESSED / "note_concepts.csv"
    if nc_path.exists():
        df = pd.read_csv(nc_path)
        id_col = "client_id" if "client_id" in df.columns else "note_id"
        rows = df[df[id_col].astype(str) == str(client_id)]
        export["concepts"] = rows.to_dict("records")

    # client profiles
    for fname in ["client_profiles.csv", "client_profiles_with_predictions.csv"]:
        fpath = DATA_OUTPUTS / fname
        if fpath.exists():
            import numpy as np
            df = pd.read_csv(fpath)
            id_col = "client_id" if "client_id" in df.columns else "id"
            rows = df[df[id_col].astype(str) == str(client_id)].replace({np.nan: None})
            if not rows.empty:
                export["profile"] = rows.to_dict("records")
                break

    # recommended actions
    ra_path = DATA_OUTPUTS / "recommended_actions.csv"
    if ra_path.exists():
        df = pd.read_csv(ra_path)
        id_col = "client_id" if "client_id" in df.columns else "id"
        rows = df[df[id_col].astype(str) == str(client_id)]
        export["actions"] = rows.to_dict("records")

    has_data = any(
        export.get(k) for k in ("notes", "concepts", "profile", "actions")
    )
    if not has_data:
        raise HTTPException(status_code=404, detail=f"No data found for client {client_id}.")

    export["exported_at"] = datetime.now().isoformat()
    return export


@app.get("/api/rgpd/config")
async def rgpd_config():
    """Return current RGPD/GDPR compliance configuration."""
    from server.shared.config import ENABLE_ANONYMIZATION, ANONYMIZATION_AGGRESSIVE
    return {
        "anonymization_enabled": ENABLE_ANONYMIZATION,
        "aggressive_mode": ANONYMIZATION_AGGRESSIVE,
        "article9_detection": True,
        "right_to_erasure_endpoint": "/api/rgpd/erase/{client_id}",
        "data_portability_endpoint": "/api/rgpd/export/{client_id}",
        "audit_endpoint": "/api/rgpd/audit",
    }


def main():
    """Start the API server."""
    parser = argparse.ArgumentParser(description="LVMH API Server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting LVMH API Server on {args.host}:{args.port}")
    logger.info("Dashboard can connect via:")
    logger.info(f"  - Local: http://localhost:{args.port}")
    logger.info(f"  - Network: http://{args.host}:{args.port}")
    
    uvicorn.run(
        "server.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
