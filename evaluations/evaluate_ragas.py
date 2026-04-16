import json
import os
from dotenv import load_dotenv
from datasets import load_from_disk
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextRecall,
    ContextPrecision
)
from ragas.llms import llm_factory
from openai import OpenAI
from rag.embeddings import embedding

from langchain_openai import ChatOpenAI


load_dotenv()

# client = OpenAI(
#     api_key=os.getenv("QWEN_API_KEY"),
#     base_url=os.getenv("QWEN_BASE_URL"),
# )

# qwen_llm = llm_factory(
#     model=os.getenv("MODEL_NAME"),
#     client=client,
#     temperature=0.1,
#     max_tokens=2048,
#     # model_kwargs={"extra_body": {"enable_thinking": False}}
#     extra_body={"enable_thinking": False}
# )



qwen_llm=ChatOpenAI(
            model=os.getenv("MODEL_NAME"),
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL"),
            temperature=0.1,
            max_tokens=1024,
            model_kwargs={"extra_body": {"enable_thinking": False}}
        )
qwen_embedding = embedding

    
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

def run_ragas_evaluation():
    dataset = load_from_disk("evaluations/rag_eval_dataset")
    print("\n===== 开始计算RAGAS评估指标 =====")

    result = evaluate(
        llm=qwen_llm,
        embeddings=qwen_embedding,
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerCorrectness(),
            ContextRecall(),
            ContextPrecision()
        ]
    )
    print(result)
    with open("evaluations/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(dict(result), f, ensure_ascii=False, indent=2)
        print("\n评估结果已保存到 evaluations/evaluation_results.json")

if __name__ == "__main__":
    run_ragas_evaluation()