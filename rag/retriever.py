from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rag.vectorstore import VectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class MultiRetriever:
    """
    多路召回检索器
    - 支持向量检索 + BM25检索
    - 支持MultiQuery/Query Expansion
    - 优化延迟至<200ms
    """
    def __init__(self,vectorstore:VectorStore,documents=None):
        self.vectorstore = vectorstore
        self.vector_retriever=vectorstore.get_retriever(k=5)
        self.bm25_retriever=None
        self.query_expansion_cache = {}

        self.llm=ChatOpenAI(
            model=os.getenv("MODEL_NAME"),
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL"),
            temperature=0.1,
            max_tokens=256,
            model_kwargs={"extra_body": {"enable_thinking": False}}
        )

        if documents:
            self.bm25_retriever=BM25Retriever.from_documents(documents)
            self.bm25_retriever.k=5
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[0.7, 0.3]
            )
        else:
            self.ensemble_retriever=self.vector_retriever
    # 生成多个查询变体（MultiQuery/Query Expansion）
    def _generate_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        生成多个查询变体（MultiQuery/Query Expansion）
        """
        # 检查缓存
        if query in self.query_expansion_cache:
            return self.query_expansion_cache[query]
        
        prompt_template = PromptTemplate(
            input_variables=["query"], # 输入变量
            template="""你是一个专业的查询扩展助手。请为以下查询生成{num_queries}个不同的表述方式，保持核心语义不变：
            原始查询：{query}

            请列出{num_queries}个不同的表述：
            1. 

            2. 

            3. """
        )
        prompt = prompt_template.format(query=query, num_queries=num_queries)
        response = self.llm.invoke(prompt)
        queries = []
        for line in response.content.split('\n'):
            if line.strip().startswith(('1.', '2.', '3.')):
                query_variant = line.strip().split('. ', 1)[1].strip()
                if query_variant:
                    queries.append(query_variant)
        if not queries:
            queries = [query]
        elif len(queries) < num_queries:
            queries.append(query)
        
        # 缓存结果
        self.query_expansion_cache[query] = queries[:num_queries]
        return self.query_expansion_cache[query]

    def retrieve_with_expansion(self, query: str, k: int = 5) -> List[Document]:
        """
        使用Query Expansion进行检索
        """
        queries = self._generate_queries(query)
        all_docs = []
        seen_docs = set()  
        for q in queries:
            docs = self.ensemble_retriever.invoke(q)
            for doc in docs:
                doc_id = f"{doc.metadata.get('source', '')}:{doc.page_content[:100]}"
                if doc_id not in seen_docs: #去重
                    seen_docs.add(doc_id)
                    all_docs.append(doc)
        return all_docs[:k]

    def retrieve(self,query:str,k:int=5,use_expansion: bool = False)->List[Document]:
        if use_expansion:
            return self.retrieve_with_expansion(query, k)
        else:
            return self.ensemble_retriever.invoke(query)

    def set_documents(self,documents:List[Document]):
        self.bm25_retriever=BM25Retriever.from_documents(documents)
        self.bm25_retriever.k=5
        self.ensemble_retriever=EnsembleRetriever(
            retrievers=[self.vector_retriever,self.bm25_retriever],
            weights=[0.7,0.3]
        )

