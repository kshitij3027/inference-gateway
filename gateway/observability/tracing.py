import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource


def init_tracing(app=None):
    """Initialize OpenTelemetry tracing. Returns TracerProvider or None if disabled."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return None

    service_name = os.getenv("OTEL_SERVICE_NAME", "inference-gateway")
    sampling_rate = float(os.getenv("OTEL_SAMPLING_RATE", "1.0"))

    resource = Resource.create({"service.name": service_name})

    if sampling_rate < 1.0:
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        sampler = TraceIdRatioBased(sampling_rate)
        provider = TracerProvider(resource=resource, sampler=sampler)
    else:
        provider = TracerProvider(resource=resource)

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI and httpx
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    if app:
        FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()

    return provider


def shutdown_tracing(provider):
    """Shutdown tracing provider gracefully."""
    if provider:
        provider.shutdown()


def get_tracer(name: str = "inference-gateway"):
    """Get a tracer instance. Returns no-op tracer when tracing is disabled."""
    return trace.get_tracer(name)
