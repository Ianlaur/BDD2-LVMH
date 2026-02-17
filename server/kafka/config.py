"""
Kafka configuration — reads from environment / .env file.

Supports:
    - Local Kafka / Redpanda (KAFKA_BOOTSTRAP_SERVERS=localhost:9092)
    - Confluent Cloud        (KAFKA_BOOTSTRAP_SERVERS=pkc-xxx.eu-central-1.aws.confluent.cloud:9092)
    - Any Kafka-compatible broker

Set KAFKA_BOOTSTRAP_SERVERS to enable Kafka. If unset, Kafka is disabled
and the pipeline runs directly in-process (legacy mode).
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from dotenv import load_dotenv
    _env_path = _PROJECT_ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Kafka settings
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "")
KAFKA_SECURITY_PROTOCOL: str = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SASL_MECHANISM: str = os.environ.get("KAFKA_SASL_MECHANISM", "")
KAFKA_SASL_USERNAME: str = os.environ.get("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD: str = os.environ.get("KAFKA_SASL_PASSWORD", "")

KAFKA_ENABLED: bool = bool(KAFKA_BOOTSTRAP_SERVERS.strip())

# Topic names
TOPIC_UPLOADS = "lvmh.uploads"
TOPIC_PIPELINE_STATUS = "lvmh.pipeline.status"
TOPIC_PIPELINE_RESULTS = "lvmh.pipeline.results"
TOPIC_AUDIT = "lvmh.audit"

ALL_TOPICS = [TOPIC_UPLOADS, TOPIC_PIPELINE_STATUS, TOPIC_PIPELINE_RESULTS, TOPIC_AUDIT]

# Consumer group
CONSUMER_GROUP = "lvmh-pipeline-workers"


def get_common_config() -> dict:
    """Return the base config dict shared by producer and consumer."""
    cfg: dict = {
        "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS,
    }

    if KAFKA_SECURITY_PROTOCOL != "PLAINTEXT":
        cfg["security_protocol"] = KAFKA_SECURITY_PROTOCOL

    if KAFKA_SASL_MECHANISM:
        cfg["sasl_mechanism"] = KAFKA_SASL_MECHANISM
        cfg["sasl_plain_username"] = KAFKA_SASL_USERNAME
        cfg["sasl_plain_password"] = KAFKA_SASL_PASSWORD

    return cfg
