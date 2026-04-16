import os
import torch
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from typing import List

class Reranker:
    """
    重排器，使用BGE-Reranker提升检索准确率
    优化延迟至<200ms
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name="E:/myRoad/RAGWise/bge-reranker-large"
        )
        self.rerank_cache = {}
        # 初始化一次 compressor，避免每次创建
        self.compressor = CrossEncoderReranker(
            model=self.reranker_model,
        )

    def rerank_documents(self,query:str,documents:List[Document],top_k=3)->List[Document]:
        # 生成缓存键
        def generate_cache_key(query, documents):
            doc_signatures = [f"{doc.metadata.get('source', '')}:{doc.page_content[:100]}" for doc in documents]
            return f"{query}:{':'.join(doc_signatures)}"
        
        cache_key = generate_cache_key(query, documents)
        
        # 检查缓存
        if cache_key in self.rerank_cache:
            return self.rerank_cache[cache_key][:top_k]
        
        if len(documents) > 10:
            documents=documents[:10]
        
        # 使用预初始化的 compressor
        compressed_docs=self.compressor.compress_documents(
            query=query,
            documents=documents
        )
        
        # 缓存结果
        self.rerank_cache[cache_key] = compressed_docs
        return compressed_docs[:top_k]

    def rerank_with_score(self, query: str, documents: List[Document], top_k: int = 3) -> List[tuple]:
        # 生成缓存键
        def generate_cache_key(query, documents):
            doc_signatures = [f"{doc.metadata.get('source', '')}:{doc.page_content[:100]}" for doc in documents]
            return f"{query}:{':'.join(doc_signatures)}:with_score"
        
        cache_key = generate_cache_key(query, documents)
        
        # 检查缓存
        if cache_key in self.rerank_cache:
            return self.rerank_cache[cache_key][:top_k]
        
        if len(documents) > 10:
            documents = documents[:10]
        
        scores = self.reranker_model.predict([(query, doc.page_content) for doc in documents])
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 缓存结果
        self.rerank_cache[cache_key] = doc_score_pairs
        return doc_score_pairs[:top_k]
        