import json
import os
from dotenv import load_dotenv
from datasets import Dataset

from rag.vectorstore import VectorStore
from rag.retriever import MultiRetriever
from rag.reranker import Reranker
from rag.chain import RAGChain

load_dotenv()

def load_test_set():
    with open("evaluations/test_set.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]

def generate_rag_dataset():
    vectorstore = VectorStore()
    reranker = Reranker()
    retriever = MultiRetriever(vectorstore)
    rag_chain = RAGChain(retriever, reranker)

    test_cases = load_test_set()

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        context = test_case["context"]

        print(f"\n生成第 {i} 条数据: {question}")
        try:
            answer = rag_chain.run(question)
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
    dataset.save_to_disk("evaluations/rag_eval_dataset")
    print(f"\n===== 数据集生成完成！已保存至 evaluations/rag_eval_dataset =====")

if __name__ == "__main__":
    generate_rag_dataset()