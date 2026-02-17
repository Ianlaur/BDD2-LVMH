"""
Kafka async producer for the LVMH pipeline.

Usage:
    from server.kafka.producer import produce_event
    await produce_event("lvmh.uploads", {"type": "voice_memo", "client_id": "VM001", ...})
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from server.kafka.config import KAFKA_ENABLED, get_common_config

logger = logging.getLogger(__name__)

_producer = None


async def _get_producer():
    """Lazy-init the aiokafka producer."""
    global _producer
    if _producer is None:
        from aiokafka import AIOKafkaProducer

        cfg = get_common_config()
        _producer = AIOKafkaProducer(
            **cfg,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retry_backoff_ms=200,
            request_timeout_ms=10_000,
        )
        await _producer.start()
        logger.info("Kafka producer started")
    return _producer


async def produce_event(
    topic: str,
    payload: dict,
    key: Optional[str] = None,
) -> bool:
    """
    Publish a JSON event to a Kafka topic.

    Args:
        topic:   Kafka topic name
        payload: Dict to JSON-serialize as the message value
        key:     Optional partition key (e.g. client_id)

    Returns:
        True if sent, False if Kafka is disabled or send failed.
    """
    if not KAFKA_ENABLED:
        return False

    # Stamp every event with metadata
    payload.setdefault("_ts", datetime.now(timezone.utc).isoformat())
    payload.setdefault("_topic", topic)

    try:
        producer = await _get_producer()
        await producer.send_and_wait(topic, value=payload, key=key)
        logger.debug(f"Kafka → {topic}: {key or '(no key)'}")
        return True
    except Exception as e:
        logger.error(f"Kafka produce failed [{topic}]: {e}")
        return False


async def close_producer():
    """Flush and close the producer (call on app shutdown)."""
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
        logger.info("Kafka producer closed")
