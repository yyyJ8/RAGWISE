from typing import List, Optional
from langchain_core.documents import Document
import os
import tempfile
from rag.loader import DocumentLoader
from rag.splitter import DocumentSplitter
from rag.vectorstore import VectorStore
from rag.retriever import MultiRetriever
from rag.reranker import Reranker
from rag.chain import RAGChain

class RAGService:
    def __init__(self):
        self.vectorstore = VectorStore()
        self.reranker = Reranker()
        self.retriever = MultiRetriever(self.vectorstore)
        self.rag_chain = RAGChain(self.retriever, self.reranker)
        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter()
    
    def process_documents(self,file_path:List[str])->dict:
        documents=[]
        try:
            docs=self.loader.load_documents(file_path,show_progress=False)
            for doc in docs:
                chunks=self.splitter.split_document(doc.page_content)
                for i,chunk in enumerate(chunks):
                    chunk_doc=type(doc)(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_id":i
                        }
                    )
                    documents.append(chunk_doc)
            count=self.vectorstore.add_documents(documents)
            self.retriever.set_documents(documents)
            return {
                "status": "success",
                "message": f"成功处理 {len(file_path)} 个文件，生成 {count} 个文档块",
                "file_count": len(file_path),
                "chunk_count": count
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"处理文档时出错: {str(e)}",
                "file_count": 0,
                "chunk_count": 0
            }
    
    def query(self, question: str) -> str:
        return self.rag_chain.run(question)

    def stream_chat(self, question: str):
        return self.rag_chain.stream(question)

    def get_vectorstore_status(self) -> dict:
        return {
            "status": "active",
            "service": "RAGService"
        } 
rag_service = RAGService()