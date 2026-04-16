from typing import List,Dict,Any
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)
from langchain_core.documents import Document
from tqdm import tqdm
import os

class DocumentLoader:
    """
    多格式文档加载器，支持PDF、Word、Markdown、TXT等格式
    """
    SUPPORTED_EXTENSIONS={
        ".pdf":PyPDFLoader,
        ".docx":UnstructuredWordDocumentLoader,
        ".md":UnstructuredMarkdownLoader,
        ".txt":TextLoader,
    }

    @classmethod
    def load_single_document(cls,file_path:Path)->List[Document]:
        """加载单个文档"""
        file_path=Path(file_path)
        ext=file_path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式：{ext}")
        loader_class=cls.SUPPORTED_EXTENSIONS[ext]
        if ext == ".pdf":
            loader=loader_class(str(file_path))
        else:
            loader=loader_class(str(file_path),encoding="utf-8-sig")

        docs=loader.load()
        for doc in docs:
            doc.metadata["source"]=str(file_path)
            doc.metadata["file_name"]=file_path.name
            doc.metadata["file_size"]=file_path.stat().st_size
            if ext==".pdf" and "page" not in doc.metadata:
                doc.metadata["page"]=1
            doc.metadata["upload_time"] = os.path.getmtime(str(file_path))
        return docs

    @classmethod
    def load_documents(cls,file_paths:List[str],show_progress:bool=True)->List[Document]:
        """批量加载多个文档，支持进度条"""
        all_docs=[]
        iterator=tqdm(file_paths,desc="加载文档") if show_progress else file_paths
        for file_path in iterator:
            try:
                docs=cls.load_single_document(file_path)
                all_docs.extend(docs)
                if show_progress:
                    iterator.set_postfix({"已加载": len(all_docs), "文件": os.path.basename(file_path)})
            except Exception as e:
                print(f"加载文件 {file_path} 时出错：{e}")
        return all_docs

    @classmethod
    def load_from_directory(cls,directory: str,extensions: List[str] = None)->List[Document]:
        """从目录加载所有支持的文档"""
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在：{directory_path}")
        if extensions is None:
            extensions=cls.SUPPORTED_EXTENSIONS.keys()
        file_paths = []
        for ext in extensions:
            file_paths.extend(directory_path.glob(f"*{ext}"))
        return cls.load_documents([str(fp) for fp in file_paths])
   