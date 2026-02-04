"""
FastAPI server for LVMH Dashboard - Remote access to data and pipeline execution.

Usage:
    python -m server.api_server
    
    Or with custom host/port:
    python -m server.api_server --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import json
import argparse
import uvicorn
from typing import Optional
import logging
import shutil
from datetime import datetime

from server.shared.config import BASE_DIR, DATA_OUTPUTS, TAXONOMY_DIR, DATA_INPUT
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
