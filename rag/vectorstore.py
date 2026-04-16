from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag.embeddings import embedding
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()


class VectorStore:
    def __init__(self, collection_name="rag_collection"):
        self.persistence_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")  # 添加默认值
        self.embeddings = embedding
        self.collection_name = collection_name
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persistence_directory
        )
        
    # 添加文档到向量库
    def add_documents(self, documents: List[Document], **kwargs):
        self.vectorstore.add_documents(documents, **kwargs)
        return len(documents)

    # 文本到向量库 返回IDs
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None) -> List[str]:
        return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)

    # 查询向量库
    def query(self, query: str, k: int = 5, filter: Optional[Dict] = None, **kwargs) -> List[Document]:  # 添加类型提示
        return self.vectorstore.similarity_search(query, k=k, filter=filter, **kwargs)

    # 根据向量查询文档
    def query_by_vector(self, embedding_vector: List[float], k: int = 5, filter: Optional[Dict] = None, **kwargs) -> List[Document]:
        return self.vectorstore.similarity_search_by_vector(embedding_vector, k=k, filter=filter, **kwargs)
              
    # 搜索向量库，返回【文档 + 相似度分数】
    def search_with_score(self, query: str, k: int = 5, filter: Optional[Dict] = None, **kwargs) -> List[tuple]:  # 添加类型提示
        return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter, **kwargs)

    # 获取检索器
    def get_retriever(self, k: int = 5):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    # 更新文档
    def update_document(self, document_id: str, document: Document):
        self.vectorstore.update_document(document_id, document)
        

    # 批量更新文档
    def update_documents(self, ids: List[str], documents: List[Document]):
        for i, doc_id in enumerate(ids):
            self.vectorstore.update_document(doc_id, documents[i])
        

    # 删除文档
    def delete_document(self, document_id: str):
        self.vectorstore.delete(ids=[document_id])  # 修正参数名
        

    # 批量删除文档
    def delete_documents(self, document_ids: List[str]):
        self.vectorstore.delete(ids=document_ids)
        

    # 根据筛选条件删除文档
    def delete_by_filter(self, filter_dict: Dict):
        self.vectorstore.delete(filter=filter_dict)
        

    # 获取集合信息
    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "name": self.collection_name,
            "count": self.vectorstore._collection.count(),
            "persist_directory": self.persistence_directory
        }

    # 清空集合
    def clear_collection(self):
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persistence_directory
        )