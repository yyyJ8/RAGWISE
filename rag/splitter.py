from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import DashScopeEmbeddings
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()


class DocumentSplitter:
    BEST_PRACTICE_CONFIG = {
        "chunk_size": 512,  # tokens
        "chunk_overlap": 50,  # tokens
        "min_chunk_size": 100,
        "max_chunk_size": 1024
    }
    def __init__(self): 
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=os.getenv("QWEN_API_KEY")
        )

    def split_with_recursive(self, text, chunk_size=512, chunk_overlap=50): 
        """使用RecursiveCharacterTextSplitter进行分块"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n", # 段落
                "\n", # 行
                ".", "!", "?", #句子
                ",", " ", #短语/空格
                ""
            ],
            length_function=len
        )
        chunks = splitter.split_text(text)
        return chunks

    def split_with_semantic(self, text,breakpoint_threshold_amount: int = 95):
        """使用SemanticChunker进行语义分块"""
        splitter = SemanticChunker(self.embeddings,breakpoint_threshold_amount=breakpoint_threshold_amount)
        chunks = splitter.split_text(text)
        return chunks

    def split_with_character(self, text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
        """使用CharacterTextSplitter进行分块"""
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
        chunks = splitter.split_text(text)
        return chunks

    def split_document(self, text, strategy="recursive", chunk_size=512, chunk_overlap=50,breakpoint_threshold: int = 95):
        """根据策略选择分块方式"""
        if strategy == "semantic":
            return self.split_with_semantic(text,breakpoint_threshold)
        elif strategy=="character":
            return self.split_with_character(text, chunk_size, chunk_overlap)
        else:
            return self.split_with_recursive(text, chunk_size, chunk_overlap)

    def compare_strategies(self, text: str) -> Dict[str, Dict[str, any]]:
        """对比不同分块策略的效果"""
        results={}
        recursive_chunks=self.split_with_recursive(text,512,50)
        results["recursive"]={
            "strategy":"recursive",
            "chunk_count":len(recursive_chunks),
            "avg_chunk_size":sum(len(c) for c in recursive_chunks)/len(recursive_chunks) if recursive_chunks else 0,
            "chunk":recursive_chunks
        }
        semantic_chunks=self.split_with_semantic(text)
        results["semantic"]={
            "strategy":"semantic",
            "chunk_count":len(semantic_chunks),
            "avg_chunk_size":sum(len(c) for c in semantic_chunks)/len(semantic_chunks) if semantic_chunks else 0,
            "chunk":semantic_chunks
        }
        character_chunks=self.split_with_character(text, chunk_size=512, chunk_overlap=50)
        results["character"]={
            "strategy":"character",
            "chunk_count":len(character_chunks),
            "avg_chunk_size":sum(len(c) for c in character_chunks)/len(character_chunks) if character_chunks else 0,
            "chunk":character_chunks
        }
        return results
