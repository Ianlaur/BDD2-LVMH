"""
FastAPI server for LVMH Dashboard - Remote access to data and pipeline execution.
Now with Neon PostgreSQL for persistent storage and multi-user support.

Usage:
    python -m server.api_server
    
    Or with custom host/port:
    python -m server.api_server --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Depends, Body, Request
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
from contextlib import asynccontextmanager

from server.shared.config import BASE_DIR, DATA_OUTPUTS, TAXONOMY_DIR, DATA_INPUT, DATA_PROCESSED
from server.run_all import run_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB availability flag — if DATABASE_URL is not set, fall back to file-based
# ---------------------------------------------------------------------------
# Load .env FIRST, before checking DATABASE_URL
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env, override=True)
        logger.info(f"Loaded .env from {_env}")
except ImportError:
    pass

DB_AVAILABLE = bool(os.environ.get("DATABASE_URL", "").strip("'\""))

# Kafka availability — if KAFKA_BOOTSTRAP_SERVERS is set, use event-driven mode
try:
    from server.kafka.config import KAFKA_ENABLED
except ImportError:
    KAFKA_ENABLED = False

# ---------------------------------------------------------------------------
# App lifecycle: init DB + Kafka on startup, close on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    if DB_AVAILABLE:
        try:
            from server.db.schema import init_database
            from server.db.connection import get_async_pool
            init_database()  # create tables if needed (sync)
            await get_async_pool()  # warm up async pool
            logger.info("✅ Database connected and schema initialized")
        except Exception as e:
            logger.warning(f"⚠️  Database init failed: {e}. Falling back to file-based mode.")
    else:
        logger.info("ℹ️  No DATABASE_URL set — running in file-only mode")

    # Start Kafka producer if configured
    if KAFKA_ENABLED:
        try:
            from server.kafka.producer import _get_producer
            await _get_producer()
            logger.info("✅ Kafka producer connected")
        except Exception as e:
            logger.warning(f"⚠️  Kafka producer init failed: {e}. Running without Kafka.")
    else:
        logger.info("ℹ️  No KAFKA_BOOTSTRAP_SERVERS set — running without Kafka")

    yield  # app runs here

    # Shutdown
    if KAFKA_ENABLED:
        try:
            from server.kafka.producer import close_producer
            await close_producer()
        except Exception:
            pass

    if DB_AVAILABLE:
        try:
            from server.db.connection import close_async_pool, close_sync_connection
            await close_async_pool()
            close_sync_connection()
        except Exception:
            pass


app = FastAPI(
    title="LVMH Voice-to-Tag API",
    description="API for accessing LVMH client intelligence data and running the pipeline",
    version="2.0.0",
    lifespan=lifespan,
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
        "version": "2.0.0",
        "database": "connected" if DB_AVAILABLE else "file-only",
        "kafka": "connected" if KAFKA_ENABLED else "disabled",
        "endpoints": {
            "data": "/api/data",
            "pipeline_status": "/api/pipeline/status",
            "run_pipeline": "/api/pipeline/run",
            "events_sse": "/api/events",
            "outputs": "/api/outputs/{filename}",
            "auth": "/api/auth/login",
            "users": "/api/users",
        }
    }


@app.get("/api/data")
@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """
    Get the main dashboard data.
    If DB is available, reads from Neon PostgreSQL.
    Otherwise falls back to data.json on disk.
    """
    # --- DB path ---
    if DB_AVAILABLE:
        try:
            from server.db.crud import async_get_dashboard_data
            data = await async_get_dashboard_data()
            if data.get("clients"):
                return data
            # If DB has no clients yet, fall through to file
            logger.info("DB has no clients — falling back to data.json")
        except Exception as e:
            logger.warning(f"DB read failed, falling back to file: {e}")

    # --- File path (fallback) ---
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
    # If DB is available, use the full 360° view
    if DB_AVAILABLE:
        try:
            from server.db.crud import async_get_client_360
            result = await async_get_client_360(client_id)
            if result:
                return result
        except Exception as e:
            logger.warning(f"DB client 360 failed: {e}, falling back to file-based")

    # Fallback to file-based
    clients = await get_clients()
    for client in clients:
        if str(client.get('client_id')) == client_id or str(client.get('id')) == client_id:
            return client
    raise HTTPException(status_code=404, detail=f"Client {client_id} not found")


@app.get("/api/kpi")
async def get_kpi_data():
    """Get executive KPI dashboard data."""
    if DB_AVAILABLE:
        try:
            from server.db.crud import async_get_kpi_data
            return await async_get_kpi_data()
        except Exception as e:
            logger.error(f"KPI data fetch failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=503, detail="Database required for KPI data")


@app.post("/api/scores/compute")
async def compute_scores():
    """Compute engagement + value scores for all clients."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database required")
    try:
        from server.db.crud import async_compute_client_scores
        result = await async_compute_client_scores()
        return result
    except Exception as e:
        logger.error(f"Score computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scores")
async def get_scores(tier: str = None, sort_by: str = "overall_score",
                     limit: int = 50, offset: int = 0):
    """Get client scores with optional tier filter and sorting."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database required")
    try:
        from server.db.crud import async_get_client_scores
        return await async_get_client_scores(tier=tier, sort_by=sort_by,
                                              limit=limit, offset=offset)
    except Exception as e:
        logger.error(f"Score fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    return {**pipeline_status, "kafka_enabled": KAFKA_ENABLED}


@app.get("/api/events")
async def pipeline_events():
    """
    Server-Sent Events (SSE) endpoint for real-time pipeline status.

    If Kafka is enabled, consumes from lvmh.pipeline.status topic and
    streams events to the dashboard. If Kafka is disabled, falls back to
    polling pipeline_status dict.

    Dashboard connects via:
        const es = new EventSource('/api/events')
        es.onmessage = (e) => console.log(JSON.parse(e.data))
    """
    from starlette.responses import StreamingResponse
    import asyncio

    async def event_stream():
        if KAFKA_ENABLED:
            # Stream from Kafka pipeline status topic
            try:
                from aiokafka import AIOKafkaConsumer
                from server.kafka.config import (
                    TOPIC_PIPELINE_STATUS, get_common_config,
                )

                cfg = get_common_config()
                consumer = AIOKafkaConsumer(
                    TOPIC_PIPELINE_STATUS,
                    **cfg,
                    group_id="lvmh-sse-dashboard",
                    value_deserializer=lambda m: m.decode("utf-8"),
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                )
                await consumer.start()
                try:
                    async for msg in consumer:
                        yield f"data: {msg.value}\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    await consumer.stop()
            except Exception as e:
                # If Kafka connection fails, fall back to polling
                logger.warning(f"SSE Kafka consumer failed: {e}, falling back to polling")
                while True:
                    yield f"data: {json.dumps(pipeline_status)}\n\n"
                    await asyncio.sleep(2)
        else:
            # No Kafka — poll pipeline_status dict every 2 seconds
            while True:
                yield f"data: {json.dumps(pipeline_status)}\n\n"
                await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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


@app.post("/api/upload-voice-memo")
async def upload_voice_memo(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    transcript: Optional[str] = None,
    client_id: Optional[str] = None,
    user_id: Optional[int] = None,
    language: Optional[str] = "FR",
    duration_seconds: Optional[int] = 0,
    run_pipeline_after: bool = True,
):
    """
    Upload a voice memo, convert the transcript into a pipeline-compatible CSV,
    run the pipeline, and sync results to the Neon database.

    Parameters:
    - audio: Audio file (webm, mp3, wav, etc.)
    - transcript: Text transcript of the audio (from browser SpeechRecognition)
    - client_id: Optional client ID — auto-generated if empty
    - user_id: User who recorded this (from auth)
    - language: Language code (FR, EN, etc.)
    - duration_seconds: Recording length in seconds
    - run_pipeline_after: Whether to trigger the pipeline automatically
    """
    import pandas as pd

    if not transcript or not transcript.strip():
        raise HTTPException(
            status_code=400,
            detail="A transcript is required. Please speak into the microphone before uploading."
        )

    # --- Save audio file ---
    voice_memos_dir = DATA_INPUT / "voice_memos"
    voice_memos_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = Path(audio.filename).suffix or '.webm'
    audio_filename = f"voice_memo_{timestamp}{file_ext}"
    audio_path = voice_memos_dir / audio_filename

    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        logger.info(f"Voice memo saved to: {audio_path}")
    except Exception as e:
        logger.error(f"Failed to save voice memo: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {str(e)}")

    # --- Auto-generate client_id if not provided ---
    if not client_id:
        # Find next available VM*** id
        try:
            from server.db.connection import sync_cursor
            with sync_cursor() as cur:
                cur.execute("""
                    SELECT id FROM clients WHERE id LIKE 'VM%%'
                    ORDER BY id DESC LIMIT 1
                """)
                row = cur.fetchone()
                if row:
                    last_num = int(row[0].replace("VM", ""))
                    client_id = f"VM{last_num + 1:03d}"
                else:
                    client_id = "VM001"
        except Exception:
            client_id = f"VM{timestamp}"

    # --- Convert duration ---
    dur_mins = duration_seconds // 60 if duration_seconds else 0
    if dur_mins < 5:
        dur_label = "Short"
    elif dur_mins < 15:
        dur_label = "Medium"
    else:
        dur_label = "Long"

    # Estimate length from transcript word count
    word_count = len(transcript.strip().split())
    if word_count < 50:
        length_label = "Short"
    elif word_count < 150:
        length_label = "Medium"
    else:
        length_label = "Long"

    # --- Build pipeline-compatible CSV ---
    csv_data = pd.DataFrame([{
        "ID": client_id,
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Duration": dur_label,
        "Language": language.upper()[:2] if language else "FR",
        "Length": length_label,
        "Transcription": transcript.strip(),
    }])

    csv_filename = f"voice_memo_{client_id}_{timestamp}.csv"
    csv_path = DATA_INPUT / csv_filename
    csv_data.to_csv(csv_path, index=False)
    logger.info(f"Voice memo CSV created: {csv_path} (client={client_id}, {word_count} words)")

    # --- Save metadata JSON ---
    meta = {
        "timestamp": timestamp,
        "audio_file": audio_filename,
        "csv_file": csv_filename,
        "client_id": client_id,
        "user_id": user_id,
        "transcript": transcript.strip(),
        "language": language,
        "duration_seconds": duration_seconds,
        "word_count": word_count,
        "uploaded_at": datetime.now().isoformat(),
    }
    meta_path = voice_memos_dir / f"meta_{client_id}_{timestamp}.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # --- Create upload session in DB ---
    upload_session_id = None
    if DB_AVAILABLE:
        try:
            from server.db.connection import get_connection
            async with get_connection() as conn:
                upload_session_id = await conn.fetchval("""
                    INSERT INTO upload_sessions (user_id, filename, upload_type, status)
                    VALUES ($1, $2, 'voice_memo', 'uploaded')
                    RETURNING id
                """, user_id, audio_filename)
        except Exception as e:
            logger.warning(f"Could not create upload session: {e}")

    # --- Run pipeline (Kafka or direct) ---
    if run_pipeline_after and KAFKA_ENABLED:
        # Dispatch to Kafka — consumer worker handles pipeline + DB sync
        from server.kafka.events import emit_upload
        await emit_upload(
            upload_type="voice_memo",
            csv_path=str(csv_path),
            user_id=user_id,
            upload_session_id=upload_session_id,
            client_id=client_id,
            filename=audio_filename,
        )
        logger.info(f"Voice memo dispatched to Kafka: {client_id}")
        return {
            "status": "uploaded_and_queued",
            "client_id": client_id,
            "audio_file": audio_filename,
            "message": f"Voice memo for {client_id} queued for processing.",
            "pipeline_status": "queued",
            "upload_session_id": upload_session_id,
        }

    if run_pipeline_after:
        if pipeline_status["running"]:
            return {
                "status": "uploaded",
                "client_id": client_id,
                "audio_file": audio_filename,
                "message": "Voice memo saved but pipeline is busy. It will be processed on next run.",
                "pipeline_status": "busy",
                "upload_session_id": upload_session_id,
            }

        pipeline_status["running"] = True
        pipeline_status["last_error"] = None

        def run_voice_pipeline():
            """Background: run pipeline on the voice memo CSV, then sync to DB."""
            try:
                logger.info(f"Running pipeline for voice memo: {csv_path}")
                run_pipeline(csv_path=str(csv_path))
                pipeline_status["running"] = False
                pipeline_status["last_run"] = "success"
                logger.info("Voice memo pipeline completed")

                # Sync to DB
                if DB_AVAILABLE:
                    try:
                        from server.db.sync import sync_results_to_db
                        result = sync_results_to_db(
                            user_id=user_id,
                            upload_session_id=upload_session_id,
                        )
                        logger.info(f"Voice memo DB sync: {result}")

                        from server.db.connection import sync_cursor
                        with sync_cursor() as cur:
                            cur.execute("""
                                UPDATE upload_sessions
                                SET status = 'completed',
                                    records_added = %s, records_updated = %s
                                WHERE id = %s
                            """, (
                                result.get("new_clients", 0),
                                result.get("updated_clients", 0),
                                upload_session_id,
                            ))
                    except Exception as db_err:
                        logger.error(f"Voice memo DB sync failed: {db_err}")
            except Exception as e:
                logger.error(f"Voice memo pipeline failed: {e}")
                pipeline_status["running"] = False
                pipeline_status["last_error"] = str(e)
                pipeline_status["last_run"] = "error"
                if DB_AVAILABLE and upload_session_id:
                    try:
                        from server.db.connection import sync_cursor
                        with sync_cursor() as cur:
                            cur.execute("""
                                UPDATE upload_sessions SET status='failed', error_message=%s
                                WHERE id=%s
                            """, (str(e), upload_session_id))
                    except Exception:
                        pass

        background_tasks.add_task(run_voice_pipeline)

        return {
            "status": "uploaded_and_running",
            "client_id": client_id,
            "audio_file": audio_filename,
            "message": f"Voice memo for {client_id} uploaded — pipeline started.",
            "pipeline_status": "running",
            "upload_session_id": upload_session_id,
        }

    return {
        "status": "uploaded",
        "client_id": client_id,
        "audio_file": audio_filename,
        "message": f"Voice memo for {client_id} saved. Run pipeline manually to process.",
        "pipeline_status": "idle",
        "upload_session_id": upload_session_id,
    }


@app.post("/api/upload-csv")
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    run_pipeline_after: bool = True,
    user_id: Optional[int] = None
):
    """
    Upload a CSV file and optionally run the pipeline.
    With DB enabled, creates an upload session and syncs results after pipeline.
    
    Parameters:
    - file: CSV file to upload
    - run_pipeline_after: If true, automatically run pipeline after upload
    - user_id: Optional user ID (from auth) who is uploading
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

    # --- Create upload session in DB ---
    upload_session_id = None
    if DB_AVAILABLE:
        try:
            from server.db.connection import get_connection
            async with get_connection() as conn:
                upload_session_id = await conn.fetchval("""
                    INSERT INTO upload_sessions (user_id, filename, upload_type, status)
                    VALUES ($1, $2, 'csv', 'uploaded')
                    RETURNING id
                """, user_id, new_filename)
        except Exception as e:
            logger.warning(f"Could not create upload session: {e}")
    
    # --- Dispatch to Kafka if enabled ---
    if run_pipeline_after and KAFKA_ENABLED:
        from server.kafka.events import emit_upload
        await emit_upload(
            upload_type="csv",
            csv_path=str(file_path),
            user_id=user_id,
            upload_session_id=upload_session_id,
            filename=new_filename,
        )
        logger.info(f"CSV upload dispatched to Kafka: {new_filename}")
        return {
            "status": "uploaded_and_queued",
            "filename": new_filename,
            "path": str(file_path),
            "message": "File uploaded and queued for processing via Kafka.",
            "pipeline_status": "queued",
            "upload_session_id": upload_session_id,
        }

    # --- Direct pipeline execution (no Kafka) ---
    if run_pipeline_after:
        if pipeline_status["running"]:
            return {
                "status": "uploaded",
                "filename": new_filename,
                "path": str(file_path),
                "message": "File uploaded but pipeline is already running. Please try running it manually later.",
                "pipeline_status": "busy",
                "upload_session_id": upload_session_id
            }
        
        # Run pipeline with the uploaded file
        pipeline_status["running"] = True
        pipeline_status["last_error"] = None
        
        def run_pipeline_task():
            """Background task: run pipeline then sync results to DB."""
            try:
                logger.info(f"Running pipeline with uploaded file: {file_path}")
                run_pipeline(csv_path=str(file_path))
                pipeline_status["running"] = False
                pipeline_status["last_run"] = "success"
                logger.info("Pipeline completed successfully")

                # --- Sync results to DB ---
                if DB_AVAILABLE:
                    try:
                        from server.db.sync import sync_results_to_db
                        result = sync_results_to_db(
                            user_id=user_id,
                            upload_session_id=upload_session_id
                        )
                        logger.info(f"DB sync: {result}")

                        # Update upload session
                        from server.db.connection import get_sync_connection, sync_cursor
                        with sync_cursor() as cur:
                            cur.execute("""
                                UPDATE upload_sessions
                                SET status = 'completed',
                                    records_added = %s,
                                    records_updated = %s
                                WHERE id = %s
                            """, (
                                result.get("new_clients", 0),
                                result.get("updated_clients", 0),
                                upload_session_id,
                            ))
                    except Exception as db_err:
                        logger.error(f"DB sync failed (pipeline still OK): {db_err}")

            except Exception as e:
                logger.error(f"Pipeline failed: {e}")
                pipeline_status["running"] = False
                pipeline_status["last_error"] = str(e)
                pipeline_status["last_run"] = "error"

                # Update upload session on failure
                if DB_AVAILABLE and upload_session_id:
                    try:
                        from server.db.connection import sync_cursor
                        with sync_cursor() as cur:
                            cur.execute("""
                                UPDATE upload_sessions SET status = 'failed', error_message = %s
                                WHERE id = %s
                            """, (str(e), upload_session_id))
                    except Exception:
                        pass
        
        background_tasks.add_task(run_pipeline_task)
        
        return {
            "status": "uploaded_and_running",
            "filename": new_filename,
            "path": str(file_path),
            "message": "File uploaded and pipeline started",
            "pipeline_status": "running",
            "upload_session_id": upload_session_id
        }
    else:
        return {
            "status": "uploaded",
            "filename": new_filename,
            "path": str(file_path),
            "message": "File uploaded successfully. Run pipeline manually when ready.",
            "pipeline_status": "idle",
            "upload_session_id": upload_session_id
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


# ===================================================================
# Authentication & User Management
# ===================================================================

@app.post("/api/auth/login")
async def login(body: dict = Body(...)):
    """
    Authenticate a user. Returns user info + token placeholder.
    Body: { "username": "...", "password": "..." }
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not configured. Auth requires DATABASE_URL.")

    username = body.get("username", "")
    password = body.get("password", "")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    from server.db.crud import async_authenticate_user
    user = await async_authenticate_user(username, password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # In production, issue a JWT token here.
    # For now, return user info with a simple session indicator.
    return {
        "status": "success",
        "user": user,
        "message": f"Welcome, {user['displayName']}"
    }


@app.post("/api/auth/register")
async def register(body: dict = Body(...)):
    """
    Register a new user. Admin-only in production.
    Body: { "username": "...", "displayName": "...", "email": "...", "password": "...", "role": "sales" }
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not configured")

    import bcrypt
    from server.db.crud import async_create_user

    username = body.get("username", "").strip()
    display_name = body.get("displayName", "").strip()
    email = body.get("email", "").strip()
    password = body.get("password", "")
    role = body.get("role", "sales")

    if not username or not password or not display_name:
        raise HTTPException(status_code=400, detail="username, displayName, and password are required")

    if role not in ("admin", "sales", "manager", "viewer", "data-scientist", "data-analyst"):
        raise HTTPException(status_code=400, detail="Invalid role")

    # Hash password
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        user = await async_create_user(username, display_name, email, password_hash, role)
        if not user:
            raise HTTPException(status_code=409, detail="Username or email already exists")
        return {"status": "created", "user": user}
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(status_code=409, detail="Username or email already exists")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users")
async def get_users():
    """Get all users (admin endpoint)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not configured")

    from server.db.crud import async_get_users
    users = await async_get_users()
    # Sanitize: never return password hashes
    for u in users:
        u.pop("password_hash", None)
    return users


@app.get("/api/upload-history")
async def get_upload_history(user_id: Optional[int] = None, limit: int = 20):
    """Get recent upload history."""
    if not DB_AVAILABLE:
        return []

    from server.db.crud import async_get_upload_history
    return await async_get_upload_history(user_id=user_id, limit=limit)


@app.delete("/api/clients/{client_id}")
async def delete_client(client_id: str, hard_delete: bool = False):
    """
    Delete a client. Soft-delete by default (sets is_deleted=TRUE).
    Use hard_delete=true for GDPR Article 17 erasure.
    Also removes from file-based storage for consistency.
    """
    if DB_AVAILABLE:
        from server.db.crud import sync_soft_delete_client, sync_hard_delete_client
        if hard_delete:
            success = sync_hard_delete_client(client_id)
        else:
            success = sync_soft_delete_client(client_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    # Also call existing file-based GDPR erasure
    try:
        await rgpd_erase_client(client_id)
    except HTTPException:
        pass  # Client might not be in files

    return {"status": "deleted", "client_id": client_id, "hard_delete": hard_delete}


@app.get("/api/db/status")
async def db_status():
    """Check database connection status and stats."""
    if not DB_AVAILABLE:
        return {"connected": False, "mode": "file-only", "message": "DATABASE_URL not configured"}

    try:
        from server.db.connection import get_connection
        async with get_connection() as conn:
            client_count = await conn.fetchval("SELECT COUNT(*) FROM clients WHERE is_deleted = FALSE")
            segment_count = await conn.fetchval("SELECT COUNT(*) FROM segments")
            user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
            last_upload = await conn.fetchrow(
                "SELECT filename, status, created_at FROM upload_sessions ORDER BY created_at DESC LIMIT 1"
            )
            last_run = await conn.fetchrow(
                "SELECT status, total_time, completed_at FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
            )

        return {
            "connected": True,
            "mode": "neon-postgres",
            "stats": {
                "clients": client_count,
                "segments": segment_count,
                "users": user_count,
            },
            "lastUpload": dict(last_upload) if last_upload else None,
            "lastPipelineRun": dict(last_run) if last_run else None,
        }
    except Exception as e:
        return {"connected": False, "mode": "error", "message": str(e)}


@app.post("/api/db/init")
async def db_init_endpoint():
    """Manually trigger database schema initialization."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")
    try:
        from server.db.schema import init_database
        init_database()
        return {"status": "success", "message": "Database schema initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/db/sync")
async def db_sync_endpoint(user_id: Optional[int] = None):
    """
    Manually sync current file-based pipeline outputs to the database.
    Useful for initial migration or after manual pipeline runs.
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")
    try:
        from server.db.sync import sync_results_to_db
        result = sync_results_to_db(user_id=user_id)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# Calendar / Activation Events
# ===================================================================

@app.get("/api/calendar/events")
async def list_calendar_events(month: Optional[int] = None, year: Optional[int] = None,
                               status: Optional[str] = None):
    """List activation events, optionally filtered by month/year and status."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_get_events
    return await async_get_events(month=month, year=year, status=status)


@app.get("/api/calendar/events/{event_id}")
async def get_calendar_event(event_id: int):
    """Get event detail with matched client targets."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_get_event_detail
    event = await async_get_event_detail(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@app.post("/api/calendar/events")
async def create_calendar_event(request: Request):
    """
    Create a new activation event and auto-match clients.
    Body: { title, description?, event_date, event_end_date?, concepts (pipe-separated),
            channel, priority?, created_by? }
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    body = await request.json()
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    event_date_str = body.get("event_date", "").strip()
    if not event_date_str:
        raise HTTPException(status_code=400, detail="event_date is required")

    from datetime import date as date_type

    try:
        event_date = date_type.fromisoformat(event_date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="event_date must be YYYY-MM-DD")

    event_end_date = None
    if body.get("event_end_date"):
        try:
            event_end_date = date_type.fromisoformat(body["event_end_date"])
        except ValueError:
            raise HTTPException(status_code=400, detail="event_end_date must be YYYY-MM-DD")

    concepts = body.get("concepts", "").strip()
    channel = body.get("channel", "email").strip()
    priority = body.get("priority", "medium").strip()
    created_by = body.get("created_by")

    from server.db.crud import async_create_event
    event = await async_create_event(
        title=title,
        description=body.get("description", ""),
        event_date=event_date,
        event_end_date=event_end_date,
        concepts=concepts,
        channel=channel,
        priority=priority,
        created_by=created_by,
    )
    return event


@app.put("/api/calendar/events/{event_id}")
async def update_calendar_event(event_id: int, request: Request):
    """Update an activation event. If concepts change, re-matches clients."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    body = await request.json()

    # Convert date strings to date objects if present
    from datetime import date as date_type
    if "event_date" in body and body["event_date"]:
        try:
            body["event_date"] = date_type.fromisoformat(body["event_date"])
        except ValueError:
            raise HTTPException(status_code=400, detail="event_date must be YYYY-MM-DD")
    if "event_end_date" in body and body["event_end_date"]:
        try:
            body["event_end_date"] = date_type.fromisoformat(body["event_end_date"])
        except ValueError:
            raise HTTPException(status_code=400, detail="event_end_date must be YYYY-MM-DD")

    from server.db.crud import async_update_event
    event = await async_update_event(event_id, **body)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@app.put("/api/calendar/events/{event_id}/targets/{client_id}")
async def update_event_target(event_id: int, client_id: str, request: Request):
    """Update a target client's status (pending → notified → responded | skipped)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    body = await request.json()
    action_status = body.get("action_status", "").strip()
    if action_status not in ("pending", "notified", "responded", "skipped"):
        raise HTTPException(status_code=400, detail="action_status must be pending|notified|responded|skipped")

    from server.db.crud import async_update_target_status
    result = await async_update_target_status(event_id, client_id, action_status)
    if not result:
        raise HTTPException(status_code=404, detail="Target not found")
    return result


@app.put("/api/calendar/events/{event_id}/targets/{client_id}/outcome")
async def update_event_target_outcome(event_id: int, client_id: str, request: Request):
    """Record a business outcome for an activation target (visited, purchased, no_response)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")

    body = await request.json()
    outcome = body.get("outcome", "").strip()
    if outcome not in ("none", "visited", "purchased", "no_response"):
        raise HTTPException(status_code=400, detail="outcome must be none|visited|purchased|no_response")

    from server.db.crud import async_update_target_outcome
    result = await async_update_target_outcome(
        event_id, client_id, outcome,
        float(body.get("outcome_value", 0)),
        body.get("outcome_notes", "")
    )
    if not result:
        raise HTTPException(status_code=404, detail="Target not found")
    return result


@app.get("/api/activation-metrics")
async def get_activation_metrics():
    """Get conversion tracking metrics across all activation events."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_get_activation_metrics
    return await async_get_activation_metrics()


@app.delete("/api/calendar/events/{event_id}")
async def delete_calendar_event(event_id: int):
    """Delete an activation event and its targets."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_delete_event
    ok = await async_delete_event(event_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"status": "deleted", "event_id": event_id}


@app.get("/api/calendar/concepts")
async def list_available_concepts():
    """
    List all distinct concepts from the DB with client counts.
    Used by the dashboard's concept picker when creating events.
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_get_concept_list
    return await async_get_concept_list()


# ===================================================================
# PLAYBOOK TEMPLATES
# ===================================================================

@app.get("/api/playbooks")
async def get_playbooks(category: str = None):
    """Get all playbook templates, optionally filtered by category."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_get_playbooks
    return await async_get_playbooks(category)


@app.post("/api/playbooks")
async def create_playbook(body: dict):
    """Create a new playbook template."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_create_playbook
    row = await async_create_playbook(
        name=body["name"],
        description=body.get("description", ""),
        concepts=body.get("concepts", ""),
        channel=body.get("channel", "email"),
        priority=body.get("priority", "medium"),
        message_template=body.get("messageTemplate", ""),
        category=body.get("category", "custom"),
        created_by=body.get("createdBy"),
    )
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create playbook")
    return {"status": "created", "playbook": row}


@app.post("/api/playbooks/seed")
async def seed_playbooks():
    """Seed default playbook templates (idempotent — skips if data exists)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_seed_playbooks
    return await async_seed_playbooks()


@app.post("/api/playbooks/{playbook_id}/activate")
async def activate_playbook(playbook_id: int, body: dict):
    """
    Create an activation event from a playbook template.
    Body: { "eventName": "...", "eventDate": "YYYY-MM-DD", "matchLimit": 200 }
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_get_playbooks, async_create_event
    # Fetch the playbook
    playbooks = await async_get_playbooks()
    playbook = next((p for p in playbooks if p["id"] == playbook_id), None)
    if not playbook:
        raise HTTPException(status_code=404, detail="Playbook not found")

    event_name = body.get("eventName", playbook["name"])
    event_date_str = body.get("eventDate")
    if not event_date_str:
        from datetime import datetime, timedelta
        event_date_str = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

    from datetime import date as date_type
    event_date = date_type.fromisoformat(event_date_str)

    # Create event using the playbook's concepts
    result = await async_create_event(
        title=event_name,
        description=playbook.get("description", ""),
        event_date=event_date,
        concepts=playbook["concepts"],
        channel=playbook["channel"],
        priority=playbook["priority"],
    )
    result["fromPlaybook"] = playbook["name"]
    result["priority"] = playbook["priority"]
    return result


# ===================================================================
# ADVISOR ASSIGNMENT
# ===================================================================

@app.get("/api/advisors")
async def get_advisors():
    """Get all available advisors (users with sales/manager/admin role)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_get_advisors
    return await async_get_advisors()


@app.get("/api/advisors/workload")
async def get_advisor_workload():
    """Get workload breakdown per advisor: client counts, tier distribution."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_get_advisor_workload
    return await async_get_advisor_workload()


@app.put("/api/clients/{client_id}/advisor")
async def assign_advisor(client_id: str, body: dict):
    """Assign an advisor to a client. Body: { "advisorId": 1 }"""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    advisor_id = body.get("advisorId")
    if not advisor_id:
        raise HTTPException(status_code=400, detail="advisorId is required")
    from server.db.crud import async_assign_advisor
    result = await async_assign_advisor(client_id, advisor_id)
    if not result:
        raise HTTPException(status_code=404, detail="Client not found")
    return result


@app.delete("/api/clients/{client_id}/advisor")
async def unassign_advisor(client_id: str):
    """Remove advisor assignment from a client."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_unassign_advisor
    return await async_unassign_advisor(client_id)


@app.post("/api/advisors/bulk-assign")
async def bulk_assign_advisor(body: dict):
    """Assign an advisor to multiple clients. Body: { "clientIds": [...], "advisorId": 1 }"""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    client_ids = body.get("clientIds", [])
    advisor_id = body.get("advisorId")
    if not client_ids or not advisor_id:
        raise HTTPException(status_code=400, detail="clientIds and advisorId are required")
    from server.db.crud import async_bulk_assign_advisor
    return await async_bulk_assign_advisor(client_ids, advisor_id)


@app.post("/api/advisors/auto-assign")
async def auto_assign_advisors(body: dict = {}):
    """
    Auto-assign unassigned clients to available advisors.
    Body (optional): { "strategy": "round_robin" | "segment" }
    """
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    strategy = body.get("strategy", "round_robin") if body else "round_robin"
    from server.db.crud import async_auto_assign_advisors
    return await async_auto_assign_advisors(strategy)


# ===================================================================
# EXPORT & REPORTING
# ===================================================================

@app.get("/api/export/clients")
async def export_clients_csv():
    """Export all clients as CSV download."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    import csv, io
    from starlette.responses import StreamingResponse
    from server.db.crud import async_export_clients_csv

    rows = await async_export_clients_csv()
    if not rows:
        raise HTTPException(status_code=404, detail="No data to export")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (str(v) if v is not None else '') for k, v in row.items()})

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=clients_export.csv"},
    )


@app.get("/api/export/actions")
async def export_actions_csv():
    """Export all actions as CSV download."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    import csv, io
    from starlette.responses import StreamingResponse
    from server.db.crud import async_export_actions_csv

    rows = await async_export_actions_csv()
    if not rows:
        raise HTTPException(status_code=404, detail="No data to export")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (str(v) if v is not None else '') for k, v in row.items()})

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=actions_export.csv"},
    )


@app.get("/api/export/scores")
async def export_scores_csv():
    """Export all client scores as CSV download."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    import csv, io
    from starlette.responses import StreamingResponse
    from server.db.crud import async_export_scores_csv

    rows = await async_export_scores_csv()
    if not rows:
        raise HTTPException(status_code=404, detail="No data to export")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (str(v) if v is not None else '') for k, v in row.items()})

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scores_export.csv"},
    )


@app.get("/api/report/summary")
async def get_summary_report():
    """Generate an executive summary report with all key metrics."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    from server.db.crud import async_generate_summary_report
    return await async_generate_summary_report()


def main():
    """Start the API server."""
    parser = argparse.ArgumentParser(description="LVMH API Server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting LVMH API Server on {args.host}:{args.port}")
    logger.info(f"Database: {'Neon PostgreSQL' if DB_AVAILABLE else 'File-only mode'}")
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
