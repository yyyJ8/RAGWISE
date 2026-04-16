from rag.splitter import DocumentSplitter

def test_chunking_comparison():
    """测试不同分块策略的对比"""
    # 示例长文档
    sample_text = """
    RAGWise是一个企业级智能文档知识库系统。
    
    它的核心功能包括：
    1. 多格式文档加载：支持PDF、Word、Markdown、TXT等格式
    2. 智能分块：Recursive和Semantic双策略
    3. 向量检索：基于Chroma的语义搜索
    4. 多路召回：向量+BM25混合检索
    5. Rerank重排：BGE-Reranker优化结果
    
    技术栈：
    - LLM: 通义千问
    - Embedding: text-embedding-v4
    - 向量库: Chroma
    - 框架: LangChain/LCEL
    
    部署方式：
    支持Docker容器化部署，包括FastAPI后端和Gradio前端。
    
    评估工具：
    使用RAGAS和LangSmith进行量化评估。
    """
    
    splitter = DocumentSplitter()
    
    print("=" * 60)
    print("分块策略对比测试")
    print("=" * 60)
    
    results = splitter.compare_strategies(sample_text)
    
    for strategy_key, result in results.items():
        print(f"\n策略: {result['strategy']}")
        print(f"分块数量: {result['chunk_count']}")
        print(f"平均块长度: {result['avg_chunk_size']:.2f} 字符")
        print("-" * 40)
        for i, chunk in enumerate(result['chunk'], 1):
            print(f"  Chunk {i} ({len(chunk)} 字符): {chunk[:50]}...")
    
    print("\n" + "=" * 60)
    print("2026 Chunking最佳实践验证")
    print("=" * 60)
    print(f"推荐 chunk_size: 512 tokens")
    print(f"推荐 chunk_overlap: 50 tokens")
    print(f"推荐 min_chunk_size: 100 tokens")
    print(f"推荐 max_chunk_size: 1024 tokens")

def test_best_practice():
    """测试2026最佳实践配置"""
    splitter = DocumentSplitter()
    
    sample_text = """
    这是一段测试文本，用于验证2026最佳实践分块配置。
    RAGWise系统采用512 tokens的chunk_size和50 tokens的chunk_overlap，
    这是目前业界公认的最佳实践配置，能够平衡检索精度和上下文完整性。
    """
    
    print("\n" + "=" * 60)
    print("2026最佳实践分块测试")
    print("=" * 60)
    
    chunks = splitter.split_with_recursive(
        sample_text,
        chunk_size=512,
        chunk_overlap=50
    )
    
    print(f"分块数量: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  长度: {len(chunk)} 字符")
        print(f"  内容: {chunk}")

if __name__ == "__main__":
    test_chunking_comparison()
    test_best_practice()