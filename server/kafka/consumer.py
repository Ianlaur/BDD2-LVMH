"""
Kafka consumer worker for the LVMH pipeline.

Listens on `lvmh.uploads` and processes each upload event by running the
pipeline and syncing results to the Neon DB.

Publishes real-time stage progress to `lvmh.pipeline.status` so the
dashboard can show live updates via SSE.

Run standalone:
    python -m server.kafka.consumer
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone

from server.kafka.config import (
    KAFKA_ENABLED,
    CONSUMER_GROUP,
    TOPIC_UPLOADS,
    TOPIC_PIPELINE_STATUS,
    TOPIC_PIPELINE_RESULTS,
    TOPIC_AUDIT,
    get_common_config,
)

logger = logging.getLogger(__name__)


class PipelineConsumer:
    """Consumes upload events and runs the pipeline."""

    def __init__(self):
        self._consumer = None
        self._producer = None  # for publishing status events

    async def start(self):
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

        cfg = get_common_config()

        self._consumer = AIOKafkaConsumer(
            TOPIC_UPLOADS,
            **cfg,
            group_id=CONSUMER_GROUP,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            max_poll_interval_ms=600_000,  # 10 min — pipeline can be slow
        )
        await self._consumer.start()
        logger.info(f"Kafka consumer started — listening on {TOPIC_UPLOADS}")

        self._producer = AIOKafkaProducer(
            **cfg,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        await self._producer.start()

    async def stop(self):
        if self._consumer:
            await self._consumer.stop()
        if self._producer:
            await self._producer.stop()
        logger.info("Kafka consumer stopped")

    async def _publish_status(self, event_type: str, data: dict):
        """Publish a status event for dashboard SSE consumption."""
        payload = {
            "type": event_type,
            **data,
            "_ts": datetime.now(timezone.utc).isoformat(),
        }
        try:
            await self._producer.send_and_wait(
                TOPIC_PIPELINE_STATUS,
                value=payload,
                key=data.get("client_id") or data.get("upload_session_id"),
            )
        except Exception as e:
            logger.warning(f"Could not publish status: {e}")

    async def _process_upload(self, event: dict):
        """
        Process a single upload event:
        1. Run the pipeline on the CSV
        2. Sync results to DB
        3. Publish completion event
        """
        from server.run_all import run_pipeline
        from server.db.sync import sync_results_to_db

        upload_type = event.get("upload_type", "csv")
        csv_path = event.get("csv_path")
        user_id = event.get("user_id")
        upload_session_id = event.get("upload_session_id")
        client_id = event.get("client_id")

        logger.info(f"Processing upload: type={upload_type}, csv={csv_path}, user={user_id}")

        await self._publish_status("pipeline_started", {
            "upload_session_id": str(upload_session_id),
            "client_id": client_id,
            "upload_type": upload_type,
            "stage": "starting",
        })

        start_time = time.time()

        try:
            # Run pipeline (blocking — runs in thread pool)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: run_pipeline(csv_path=csv_path))

            elapsed = time.time() - start_time

            await self._publish_status("pipeline_stage", {
                "upload_session_id": str(upload_session_id),
                "client_id": client_id,
                "stage": "syncing_db",
                "elapsed": round(elapsed, 1),
            })

            # Sync to DB
            result = await loop.run_in_executor(
                None,
                lambda: sync_results_to_db(
                    user_id=user_id,
                    upload_session_id=upload_session_id,
                ),
            )

            total_time = time.time() - start_time

            # Update upload session
            try:
                from server.db.connection import sync_cursor
                with sync_cursor() as cur:
                    cur.execute("""
                        UPDATE upload_sessions
                        SET status = 'completed',
                            records_added = %s, records_updated = %s,
                            processing_time = %s
                        WHERE id = %s
                    """, (
                        result.get("new_clients", 0),
                        result.get("updated_clients", 0),
                        round(total_time, 2),
                        upload_session_id,
                    ))
            except Exception as db_err:
                logger.warning(f"Could not update upload session: {db_err}")

            # Publish completion
            completion = {
                "upload_session_id": str(upload_session_id),
                "client_id": client_id,
                "upload_type": upload_type,
                "status": "completed",
                "total_time": round(total_time, 1),
                "new_clients": result.get("new_clients", 0),
                "updated_clients": result.get("updated_clients", 0),
            }
            await self._publish_status("pipeline_completed", completion)

            try:
                await self._producer.send_and_wait(
                    TOPIC_PIPELINE_RESULTS, value=completion,
                    key=str(upload_session_id),
                )
            except Exception:
                pass

            logger.info(
                f"Upload processed: {result.get('new_clients', 0)} new, "
                f"{result.get('updated_clients', 0)} updated in {total_time:.1f}s"
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Pipeline failed for upload {upload_session_id}: {e}")

            await self._publish_status("pipeline_failed", {
                "upload_session_id": str(upload_session_id),
                "client_id": client_id,
                "status": "failed",
                "error": str(e),
                "elapsed": round(elapsed, 1),
            })

            # Update DB
            try:
                from server.db.connection import sync_cursor
                with sync_cursor() as cur:
                    cur.execute("""
                        UPDATE upload_sessions SET status='failed', error_message=%s
                        WHERE id=%s
                    """, (str(e), upload_session_id))
            except Exception:
                pass

    async def run(self):
        """Main consumer loop — runs forever."""
        await self.start()
        try:
            async for msg in self._consumer:
                event = msg.value
                logger.info(f"Received upload event: {event.get('upload_type', '?')}")
                # Process each upload — could add concurrency limit here
                await self._process_upload(event)
        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled")
        finally:
            await self.stop()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not KAFKA_ENABLED:
        logger.error(
            "KAFKA_BOOTSTRAP_SERVERS is not set. "
            "Set it in .env or environment to enable Kafka."
        )
        return

    consumer = PipelineConsumer()
    await consumer.run()


if __name__ == "__main__":
    asyncio.run(main())
