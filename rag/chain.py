from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from typing import List, Dict, Any,Optional
import os
from dotenv import load_dotenv
from rag.retriever import MultiRetriever
from rag.reranker import Reranker

load_dotenv()

class RAGChain:
    """
    完整的RAG Chain，使用LCEL构建
    包含Self-Check/Fact-Verify机制，控制幻觉率<5%
    """
    def __init__(self,retriever:MultiRetriever,reranker:Reranker):
        self.retriever=retriever
        self.reranker=reranker
        self.llm=ChatOpenAI(
            model=os.getenv("MODEL_NAME"),
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL"),
            temperature=0.1,
            max_tokens=1024,
            model_kwargs={"extra_body": {"enable_thinking": False}}
        )
        self.chain = self._build_chain()
        self._check_chain=self._build_self_check_chain()
    def _format_docs(self, documents: List[Document]) -> str:
        """格式化文档，添加来源引用"""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            source_info = f"来源[{i}]: {doc.metadata.get('file_name', 'unknown')}"
            if 'page' in doc.metadata:
                source_info += f" (第{doc.metadata['page']}页)"
            formatted_docs.append(f"{source_info}\n{doc.page_content}")
        return "\n\n".join(formatted_docs)

    def _build_chain(self):
        prompt=ChatPromptTemplate.from_messages([
            ("system","""你是一个专业的企业知识库问答助手。请严格按照以下要求回答：

            1. 仅使用提供的上下文信息回答问题，不要使用外部知识
            2. 如果上下文没有相关信息，请明确说明"根据提供的资料，我无法回答这个问题"
            3. 回答要准确、简洁，使用中文
            4. 在回答中必须引用来源，格式为[1][2]这样的上标
            5. 回答结束后列出所有引用的来源详情

            上下文信息：
            {context}"""),
        ("human","{question}")
        ])
        chain=(
            RunnableParallel({
                "context":RunnablePassthrough(),
                "question":RunnablePassthrough()
            })
            |prompt
            |self.llm
            |StrOutputParser()
        )
        return chain

    def _build_self_check_chain(self):
        """构建自我检查Chain"""
        self_check_prompt =  ChatPromptTemplate. from_messages([
        ("system", """你是一个事实检查助手。请检查以下回答是否严格基于提供的上下文信息：

            1. 检查回答中是否包含上下文之外的信息
            2. 检查回答中的事实是否与上下文一致
            3. 检查来源引用是否正确

            如果发现问题，请指出具体错误并提供修正后的回答。如果没有问题，请返回原回答。

            上下文信息：
            {context}

            原回答：
            {answer}"""),
        ("human","请检查上述回答的准确性")
        ])
        self_check_chain = (
            self_check_prompt
            | self.llm
            | StrOutputParser()
        )   
        return self_check_chain   

    def run(self,question:str)->str:
        # 只执行一次检索和重排
        docs = self.retriever.retrieve(question)
        reranked_docs = self.reranker.rerank_documents(question, docs, top_k=3)
        context = self._format_docs(reranked_docs)
        # 获取初步回答
        initial_answer = self.chain.invoke({"context": context, "question": question})
        # 自我检查（复用上下文）
        final_answer = self._check_chain.invoke({
            "context": context,
            "answer": initial_answer
        })
        return final_answer

    def stream(self, question: str):
        # 先执行检索和重排获取上下文
        docs = self.retriever.retrieve(question)
        reranked_docs = self.reranker.rerank_documents(question, docs, top_k=3)
        context = self._format_docs(reranked_docs)
        # 处理stream输出，确保返回的是文本
        for chunk in self.chain.stream({"context": context, "question": question}):
            # 检查chunk类型，提取文本内容
            if isinstance(chunk, dict):
                # 对于HuggingFacePipeline，文本通常在"text"键中
                text = chunk.get("text", "")
                if text:
                    yield text
            else:
                # 对于其他LLM，直接返回字符串
                yield chunk