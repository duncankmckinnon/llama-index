import os
from time import sleep
import phoenix as px
from datetime import datetime, timedelta

from arize.pandas.logger import Client as ArizeClient
from arize.exporter import ArizeExportClient
from arize.utils.types import Environments
from phoenix.evals import run_evals, OpenAIModel, HallucinationEvaluator, QAEvaluator, RelevanceEvaluator
from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from phoenix.trace import SpanEvaluations, DocumentEvaluations

ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")

ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
ARIZE_MODEL_NAME = os.getenv("ARIZE_MODEL_NAME")
ARIZE_MODEL_VERSION = os.getenv("ARIZE_MODEL_VERSION")

eval_model = OpenAIModel(model="gpt-4-turbo-preview", api_key=OPEN_AI_API_KEY)
arize_client = ArizeClient(api_key=ARIZE_API_KEY, space_key=ARIZE_SPACE_KEY)
export_client = ArizeExportClient(api_key=ARIZE_API_KEY)
hallucination_eval = HallucinationEvaluator(eval_model)
qa_eval = QAEvaluator(eval_model)
relevance_eval = RelevanceEvaluator(eval_model)

def export_from_arize(start_time: datetime, end_time: datetime):
    return export_client.export_model_to_df(
        space_id=ARIZE_SPACE_ID,
        model_id=ARIZE_MODEL_NAME,
        model_version=ARIZE_MODEL_VERSION,
        environment=Environments.TRACING,
        start_time=start_time,
        end_time=end_time,
    )

def evaluate_model(start_time: datetime, stop_time: datetime):
    phoenix_client = px.Client()
    qa_df = get_qa_with_reference(phoenix_client, start_time=start_time, stop_time=stop_time)
    doc_df = get_retrieved_documents(phoenix_client, start_time=start_time, stop_time=stop_time)
    hallucination_df, qa_df = run_evals(qa_df, [hallucination_eval, qa_eval], provide_explanation=True)
    relevance_df = run_evals(doc_df, [relevance_eval], provide_explanation=True)[0]
    phoenix_client.log_evaluations(
        SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_df),
        SpanEvaluations(eval_name="QA Correctness", dataframe=qa_df),
        DocumentEvaluations(eval_name="Relevance", dataframe=relevance_df),
    )
    print("✅ You have successfully logged evaluations to Phoenix")

    tds = phoenix_client.get_trace_dataset()
    spans_df = tds.get_spans_dataframe(include_evaluations=False)
    evals_df = tds.get_evals_dataframe()
    response = arize_client.log_spans(
        dataframe=spans_df,
        evals_dataframe=evals_df,
        model_id=ARIZE_MODEL_NAME,
        model_version=ARIZE_MODEL_VERSION, 
    )
    if response.status_code != 200:
        print(f"❌ logging failed with response code {response.status_code}, {response.text}")
    else:
        print(f"✅ You have successfully logged traces set to Arize")

if __name__ == "__main__":
    while True:
        current_time = datetime.now()
        start_time = (current_time - timedelta(min=2, seconds=15)).isoformat()
        stop_time = current_time.isoformat()
        evaluate_model(start_time, stop_time)
        sleep(120)
