import json
import os
import asyncio
from dotenv import load_dotenv
from datasets import Dataset

# ====================== RAGAS 0.4.3 官方正确导入（无弃用警告） ======================
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextRecall,
    ContextPrecision
)

from openai import OpenAI
from ragas.llms import llm_factory


from rag.vectorstore import VectorStore
from rag.retriever import MultiRetriever
from rag.reranker import Reranker
from rag.chain import RAGChain
from rag.embeddings import embedding
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("QWEN_API_KEY", "dummy_key_for_ragas")


client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

qwen_llm = llm_factory(
    model="qwen3.5-plus",
    client=client,
    temperature=0.1,
    max_tokens=1024
)
embedding = embedding
faithfulness = Faithfulness()
answer_correctness = AnswerCorrectness()
context_recall = ContextRecall()
context_precision = ContextPrecision()

def load_test_set():
    with open("evaluations/test_set.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


class NaiveRAG:
    def __init__(self, vectorstore):
        self.retriever = vectorstore.get_retriever(k=3)

    def run(self, question):
        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        answer = f"根据上下文信息：{context[:300]}..."
        return answer, [doc.page_content for doc in docs]


def evaluate_baseline():
    test_cases = load_test_set()
    vectorstore = VectorStore()
    naive_rag = NaiveRAG(vectorstore)

    questions, ground_truths, answers, contexts = [], [], [], []
    
    for test_case in test_cases:
        q = test_case["question"]
        gt = test_case["ground_truth"]
        ans, ctx = naive_rag.run(q)
        
        questions.append(q)
        ground_truths.append(gt)
        answers.append(ans)
        contexts.append(ctx)
    
    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_correctness, context_recall, context_precision],
        llm=qwen_llm,
        embeddings=embedding
    )
    return result

# ====================== 评估高级RAG ======================
def evaluate_advanced_rag():
    test_cases = load_test_set()
    vectorstore = VectorStore()
    retriever = MultiRetriever(vectorstore)
    reranker = Reranker()
    rag_chain = RAGChain(retriever, reranker)

    questions, ground_truths, answers, contexts = [], [], [], []
    
    for test_case in test_cases:
        q = test_case["question"]
        gt = test_case["ground_truth"]
        
        # 获取真实检索+重排上下文
        docs = rag_chain.retriever.retrieve(q)
        reranked_docs = rag_chain.reranker.rerank_documents(q, docs, top_k=3)
        ctx = [doc.page_content for doc in reranked_docs]
        ans = rag_chain.run(q)
        
        questions.append(q)
        ground_truths.append(gt)
        answers.append(ans)
        contexts.append(ctx)

    dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_correctness, context_recall, context_precision],
        llm=qwen_llm,
        embeddings=embedding
    )
    return result

# ====================== 模型对比 ======================
def compare_models():
    print("正在评估 基线RAG (简单检索)...")
    baseline_result = evaluate_baseline()
    print(f"基线RAG结果:\n{baseline_result}\n")

    print("正在评估 高级RAG (多路召回+重排+Self-Check)...")
    advanced_result = evaluate_advanced_rag()
    print(f"高级RAG结果:\n{advanced_result}\n")

    print("="*50)
    print("模型提升对比")
    print("="*50)
    improvements = {}
    for metric in baseline_result.keys():
        base = baseline_result[metric]
        adv = advanced_result[metric]
        improve = ((adv - base) / base) * 100 if base != 0 else 0
        improvements[metric] = improve
        print(f"{metric}: 基线 {base:.3f} → 高级 {adv:.3f} | +{improve:.1f}%")

    # 保存结果
    os.makedirs("evaluations", exist_ok=True)
    comparison_result = {
        "baseline": baseline_result.to_dict(),
        "advanced": advanced_result.to_dict(),
        "improvements": improvements
    }
    with open("evaluations/comparison_result.json", "w", encoding="utf-8") as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)

    print("\n✅ 对比结果已保存至: evaluations/comparison_result.json")

if __name__ == "__main__":
    # 修复Windows异步事件环报错
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    compare_models()