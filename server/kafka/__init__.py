"""
Kafka event streaming for the LVMH Voice-to-Tag pipeline.

Topics:
    lvmh.uploads          — CSV / voice memo upload events (consumed by pipeline worker)
    lvmh.pipeline.status  — real-time stage progress (consumed by dashboard SSE)
    lvmh.pipeline.results — pipeline completion results
    lvmh.audit            — user action audit trail

Optional: if KAFKA_BOOTSTRAP_SERVERS is not set, the system falls back
to direct (in-process) pipeline execution — no Kafka dependency required.
"""
