"""
监控埋点模块（OpenTelemetry + Prometheus）
提供：
  1. 自动采集 FastAPI 请求 trace 和 metric
  2. 自定义埋点上下文管理器，用于 pipeline 各环节
  3. Prometheus /metrics 端点
"""
import os
import time
from functools import wraps
from contextlib import contextmanager
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from prometheus_client import start_http_server, make_wsgi_app
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# ── 全局单例 ──
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None

# ── Prometheus 指标定义（只定义一次，避免重复注册） ──

# 请求相关
_request_count = None
_request_duration = None
_request_token_total = None
_retrieved_doc_count = None

# 各环节耗时（histogram，单位秒）
_span_duration = None


def _init_metrics(meter: metrics.Meter):
    global _request_count, _request_duration, _request_token_total
    global _retrieved_doc_count, _span_duration

    _request_count = meter.create_counter(
        name="rag_requests_total",
        description="RAG 请求总数",
        unit="1",
    )
    _request_duration = meter.create_histogram(
        name="rag_request_duration_seconds",
        description="RAG 请求总耗时（秒）",
        unit="s",
    )
    _request_token_total = meter.create_counter(
        name="rag_token_total",
        description="DeepSeek token 消耗总数",
        unit="1",
    )
    _retrieved_doc_count = meter.create_histogram(
        name="rag_retrieved_doc_count",
        description="每次请求检索到的文档数",
        unit="1",
    )
    _span_duration = meter.create_histogram(
        name="rag_span_duration_seconds",
        description="各环节耗时（秒）",
        unit="s",
        # 按环节名称和状态打标签
    )


def init_monitoring(app: FastAPI, service_name: str = "openclaw-local-rag", metrics_port: int = 8001,
                    jaeger_endpoint: Optional[str] = None):
    """
    在 FastAPI 应用上启用监控。

    调用一次即可，幂等。会额外启动一个 HTTP server 在 metrics_port 上暴露 Prometheus 指标。

    Args:
        app: FastAPI 应用实例
        service_name: 服务名称
        metrics_port: Prometheus 指标暴露端口
        jaeger_endpoint: Jaeger gRPC 端点（如 http://jaeger:4317），不为 None 时启用链路追踪
    """
    global _tracer, _meter

    if _tracer is not None:
        return  # 已初始化

    resource = Resource(attributes={SERVICE_NAME: service_name})

    # ── Trace（可同时输出到 Prometheus + Jaeger）──
    tracer_provider = TracerProvider(resource=resource)

    # 如果配置了 Jaeger 端点，添加 OTLP exporter
    jaeger_endpoint = jaeger_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if jaeger_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=jaeger_endpoint)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        print(f"📡 Jaeger 链路追踪已启用: {jaeger_endpoint}")
    else:
        print("ℹ️ Jaeger 未配置，仅本地 trace（不发送 span）")

    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(service_name)

    # ── Metric（直接暴露 Prometheus，不经过 OTLP） ──
    reader = PrometheusMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter(service_name)

    _init_metrics(_meter)

    # ── 自动插桩 FastAPI（自动记录每个请求的 trace + metric） ──
    FastAPIInstrumentor.instrument_app(app)

    # ── 额外暴露一个 /metrics 端点（也让 Prometheus server 同时监听） ──
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        from prometheus_client import generate_latest
        return PlainTextResponse(generate_latest(), media_type="text/plain")

    # 额外启动一个 Prometheus HTTP server（供容器内抓取）
    start_http_server(metrics_port)

    print(f"✅ 监控已启用: trace + metric, Prometheus 端口 {metrics_port}")


# ── 公开的辅助函数 ──

def get_tracer() -> trace.Tracer:
    """获取全局 tracer。必须在 init_monitoring 之后调用。"""
    if _tracer is None:
        return trace.get_tracer("openclaw-local-rag")
    return _tracer


def get_meter() -> metrics.Meter:
    if _meter is None:
        meter_provider = metrics.get_meter_provider()
        return meter_provider.get_meter("openclaw-local-rag")
    return _meter


@contextmanager
def span(name: str, attrs: dict = None):
    """
    对 pipeline 中一个环节进行计时 + 记录到 Prometheus。

    用法:
        with span("qdrant_search"):
            docs = await qdrant.search(query)
    """
    start = time.monotonic()
    try:
        with get_tracer().start_as_current_span(name) as current_span:
            if attrs:
                current_span.set_attributes(attrs)
            yield
    except Exception as e:
        if _span_duration:
            _span_duration.record(
                time.monotonic() - start,
                attributes={"span": name, "status": "error"},
            )
        raise
    else:
        if _span_duration:
            _span_duration.record(
                time.monotonic() - start,
                attributes={"span": name, "status": "ok"},
            )


def record_request(query: str, total_time: float, doc_count: int, token_count: int = 0):
    """记录一次完整的 RAG 请求结果。"""
    if _request_count:
        _request_count.add(1, attributes={"status": "ok"})
    if _request_duration:
        _request_duration.record(total_time)
    if _retrieved_doc_count:
        _retrieved_doc_count.record(doc_count)
    if token_count and _request_token_total:
        _request_token_total.add(token_count)


def record_error(query: str):
    """记录一次失败的 RAG 请求。"""
    if _request_count:
        _request_count.add(1, attributes={"status": "error"})
