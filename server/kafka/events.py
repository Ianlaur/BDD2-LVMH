"""
Convenience helpers for producing common event types.

These wrap produce_event() with structured payloads so callers
don't need to know the topic names or payload shapes.
"""
from typing import Optional
from server.kafka.config import (
    TOPIC_UPLOADS,
    TOPIC_PIPELINE_STATUS,
    TOPIC_AUDIT,
    KAFKA_ENABLED,
)
from server.kafka.producer import produce_event


async def emit_upload(
    upload_type: str,
    csv_path: str,
    user_id: Optional[int] = None,
    upload_session_id: Optional[int] = None,
    client_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> bool:
    """Emit an upload event for the pipeline consumer to pick up."""
    return await produce_event(
        TOPIC_UPLOADS,
        {
            "upload_type": upload_type,  # "csv" or "voice_memo"
            "csv_path": csv_path,
            "user_id": user_id,
            "upload_session_id": upload_session_id,
            "client_id": client_id,
            "filename": filename,
        },
        key=client_id or str(upload_session_id or ""),
    )


async def emit_pipeline_status(
    stage: str,
    upload_session_id: Optional[int] = None,
    client_id: Optional[str] = None,
    **extra,
) -> bool:
    """Emit a pipeline progress event (consumed by dashboard SSE)."""
    return await produce_event(
        TOPIC_PIPELINE_STATUS,
        {
            "type": "pipeline_stage",
            "stage": stage,
            "upload_session_id": str(upload_session_id) if upload_session_id else None,
            "client_id": client_id,
            **extra,
        },
        key=client_id or str(upload_session_id or ""),
    )


async def emit_audit(
    user_id: Optional[int],
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    details: Optional[dict] = None,
    ip: Optional[str] = None,
) -> bool:
    """Emit an audit event (and also write to DB for persistence)."""
    sent = await produce_event(
        TOPIC_AUDIT,
        {
            "user_id": user_id,
            "action": action,
            "target_type": target_type,
            "target_id": target_id,
            "details": details,
            "ip": ip,
        },
        key=str(user_id or "system"),
    )

    # Always write to DB as well (Kafka is for streaming, DB is source of truth)
    try:
        from server.db.crud import sync_log_audit
        sync_log_audit(
            user_id=user_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            details=details,
            ip=ip,
        )
    except Exception:
        pass  # DB write is best-effort if Kafka is primary

    return sent
