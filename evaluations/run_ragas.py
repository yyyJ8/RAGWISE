import json
import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextRecall,
    ContextPrecision
)
from datasets import Dataset
from rag.vectorstore import VectorStore
from rag.retriever import MultiRetriever
from rag.reranker import Reranker
from rag.chain import RAGChain
from ragas.llms import llm_factory
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

qwen_llm = llm_factory(
    model=os.getenv("MODEL_NAME"),
    client=client,
    temperature=0.1,
    max_tokens=256,
    model_kwargs={"extra_body": {"enable_thinking": False}}
)

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

def load_test_set():
    with open("evaluations/test_set.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]

def evaluate_rag_system():
    vectorstore = VectorStore()
    reranker = Reranker()
    retriever = MultiRetriever(vectorstore)
    rag_chain = RAGChain(retriever, reranker)
    test_cases = load_test_set()

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print("正在执行评估...")
    for i,test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        context = test_case["context"]
        print(f"\n测试 {i}: {question}")
        try:
            answer = rag_chain.run(question)
            print(f"回答: {answer[:100]}...")
        except Exception as e:
            answer = f"Error: {str(e)}"
            print(f"错误: {str(e)}")
        questions.append(question)
        ground_truths.append(ground_truth)
        answers.append(answer)
        contexts.append([context])
    
    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts
    })

    print("\n正在计算评估指标...")
    result = evaluate(
        llm=qwen_llm,
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerCorrectness(),
            ContextRecall(),
            ContextPrecision()
        ]
        #run_config={"max_workers": 15}
    )   

    print("\n评估结果:")
    print(result) 

    with open("evaluations/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print("\n评估结果已保存到 evaluations/evaluation_results.json")

if __name__ == "__main__":
    evaluate_rag_system()