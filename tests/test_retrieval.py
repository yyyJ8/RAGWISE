import time
from rag.vectorstore import VectorStore
from rag.retriever import MultiRetriever
from rag.reranker import Reranker
from langchain_core.documents import Document
import json

def test_retrieval_performance():
    """测试检索性能"""
    # 创建测试文档
    test_docs = []
    for i in range(10):
        doc = Document(
            page_content=f"这是测试文档 {i}，包含关于RAGWise系统的信息。"
            f"RAGWise是一个企业级智能文档知识库系统，支持PDF、Word、Markdown等格式。"
            f"它使用通义千问作为LLM，text-embedding-v4作为嵌入模型。"
            f"核心功能包括智能分块、多路召回、重排优化等。",
            metadata={
                "file_name": f"test_{i}.md",
                "source": f"test_{i}.md",
                "page": i + 1
            }
        )
        test_docs.append(doc)
    
    # 初始化系统
    vectorstore = VectorStore()
    vectorstore.add_documents(test_docs)
    
    retriever = MultiRetriever(vectorstore, test_docs)
    reranker = Reranker()
    
    # 测试查询
    test_queries = [
        "RAGWise支持哪些文档格式？",
        "RAGWise使用什么LLM？",
        "RAGWise的核心功能有哪些？"
    ]
    
    print("=" * 60)
    print("检索性能测试")
    print("=" * 60)
    
    total_time = 0
    for query in test_queries:
        print(f"\n查询: {query}")
        
        # 测试检索时间
        start_time = time.time()
        docs = retriever.retrieve(query, k=5)
        retrieval_time = time.time() - start_time
        
        # 测试重排时间
        start_time = time.time()
        reranked_docs = reranker.rerank_documents(query, docs, top_k=3)
        rerank_time = time.time() - start_time
        
        total_latency = retrieval_time + rerank_time
        total_time += total_latency
        
        print(f"  检索时间: {retrieval_time:.4f} 秒")
        print(f"  重排时间: {rerank_time:.4f} 秒")
        print(f"  总延迟: {total_latency:.4f} 秒")
        print(f"  延迟 < 200ms: {total_latency < 0.2}")
        
        # 显示Top-3结果
        print(f"  Top-3 结果:")
        for i, doc in enumerate(reranked_docs, 1):
            print(f"    {i}. {doc.page_content[:100]}...")
    
    avg_latency = total_time / len(test_queries)
    print("\n" + "=" * 60)
    print(f"平均延迟: {avg_latency:.4f} 秒")
    print(f"平均延迟 < 200ms: {avg_latency < 0.2}")
    print("=" * 60)

def test_query_expansion():
    """测试Query Expansion功能"""
    # 创建测试文档
    test_docs = []
    for i in range(5):
        doc = Document(
            page_content=f"这是测试文档 {i}，包含关于RAGWise系统的信息。"
            f"RAGWise是一个企业级智能文档知识库系统，支持PDF、Word、Markdown等格式。"
            f"它使用通义千问作为LLM，text-embedding-v4作为嵌入模型。"
            f"核心功能包括智能分块、多路召回、重排优化等。",
            metadata={
                "file_name": f"test_{i}.md",
                "source": f"test_{i}.md",
                "page": i + 1
            }
        )
        test_docs.append(doc)
    
    # 初始化系统
    vectorstore = VectorStore()
    vectorstore.add_documents(test_docs)
    
    retriever = MultiRetriever(vectorstore, test_docs)
    
    test_query = "RAGWise支持哪些文档格式？"
    
    print("\n" + "=" * 60)
    print("Query Expansion测试")
    print("=" * 60)
    print(f"原始查询: {test_query}")
    
    # 生成查询变体
    queries = retriever._generate_queries(test_query)
    print(f"\n生成的查询变体:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    # 测试普通检索
    normal_docs = retriever.retrieve(test_query, k=3)
    print(f"\n普通检索结果:")
    for i, doc in enumerate(normal_docs, 1):
        print(f"  {i}. {doc.page_content[:100]}...")
    
    # 测试带Expansion的检索
    expanded_docs = retriever.retrieve(test_query, k=3, use_expansion=True)
    print(f"\n带Query Expansion的检索结果:")
    for i, doc in enumerate(expanded_docs, 1):
        print(f"  {i}. {doc.page_content[:100]}...")

if __name__ == "__main__":
    test_retrieval_performance()
    test_query_expansion()