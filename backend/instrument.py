from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import os

def instrument_phoenix():
    collector_endpoint = os.getenv("COLLECTOR_ENDPOINT")
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_exporter = HTTPExporter(endpoint=collector_endpoint)
    span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument()
    print("🔭 OpenInference instrumentation enabled with Phoenix collector.")

def instrument_arize():
    arize_endpoint = os.getenv("ARIZE_ENDPOINT")
    arize_api_key = os.getenv("ARIZE_API_KEY")
    arize_space_key = os.getenv("ARIZE_SPACE_KEY")
    arize_model_id = os.getenv("ARIZE_MODEL_ID")
    arize_model_version = os.getenv("ARIZE_MODEL_VERSION")
    print(f"model id: {arize_model_id}, model version: {arize_model_version}")

    headers = f"space_key={arize_space_key},api_key={arize_api_key}"
    os.environ['OTEL_EXPORTER_OTLP_TRACES_HEADERS'] = headers
    resource = Resource(attributes={
        "model_id": arize_model_id,
        "model_version": arize_model_version,
    })
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_exporter = GRPCExporter(endpoint=arize_endpoint)
    span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument()
    print("🔭 OpenInference instrumentation enabled with Arize collector.")

def instrument():
    if os.getenv("INSTRUMENT_ARIZE"):
        instrument_arize()
    elif os.getenv("INSTRUMENT_PHOENIX"):
        instrument_phoenix()
    
    
    
