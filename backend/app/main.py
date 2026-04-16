import os
from dotenv import load_dotenv
from fastapi import FastAPI,UploadFile,File,Form,HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List,Optional
import tempfile
from rag.loader import DocumentLoader
from rag.splitter import DocumentSplitter
from rag.vectorstore import VectorStore 
from rag.retriever import MultiRetriever
from rag.reranker import Reranker
from rag.chain import RAGChain
from .schemas.schemas import UploadResponse, QueryResponse, ErrorResponse, HealthCheckResponse
load_dotenv()

app=FastAPI(
    title="RAGWise API",
    description="企业级智能文档知识库系统 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware, #跨域中间件
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

vectorstore=VectorStore()
retriever=MultiRetriever(vectorstore)
reranker=Reranker()
rag_chain=RAGChain(retriever,reranker)

def verify_api_key(api_key:str=Form(...)):
    if api_key!=os.getenv("API_KEY"):
        raise HTTPException(status_code=401,detail="Invalid API key")
    return api_key

@app.post("/upload",response_model=UploadResponse)
async def upload_documents(
    files:List[UploadFile]=File(...),
    api_key:str=Depends(verify_api_key)
):
    uploaded_files=[]
    documents=[]
    try:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(file.filename)[1]) as temp_file:
                content=await file.read()
                temp_file.write(content)
                temp_file.flush()
                uploaded_files.append(temp_file.name)
        loader=DocumentLoader()
        docs=loader.load_documents(uploaded_files,show_progress=False)
        splitter=DocumentSplitter()
        for doc in docs:
            chunks=splitter.split_document(doc.page_content)
            for i,chunk in enumerate(chunks):
                chunk_doc=type(doc)(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id":i
                    }    
                )
                documents.append(chunk_doc)
        count=vectorstore.add_documents(documents)
        retriever.set_documents(documents)
        return UploadResponse(
            status="success",
            message=f"成功上传并处理 {len(files)} 个文件，生成 {count} 个文档块",
            file_count=len(files),
            chunk_count=count
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status="error",
                error="Upload failed",
                message=str(e)
            ).model_dump()
        )
    finally:
        for file_path in uploaded_files:
            if os.path.exists(file_path):
                os.remove(file_path) 
@app.post("/query",response_model=QueryResponse)
async def query(
    questions:str=Form(...),
    api_key:str=Depends(verify_api_key)
):
    try:
        result=rag_chain.run(questions)
        return QueryResponse(
            status="success",
            answer=result,
            sources=[]  
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status="error",
                error="Query failed",
                message=str(e)
            ).model_dump()
        )

@app.post("/chat")
async def chat(
    questions:str=Form(...),
    api_key:str=Depends(verify_api_key)
):
    try:
        def generate():
            for chunk in rag_chain.stream(questions):
                yield chunk
        return StreamingResponse(generate(),media_type="text/plain")
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                status="error",
                error="Chat failed",
                message=str(e)
            ).model_dump()
        )
    
@app.get("/health",response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        service="RAGWise API"
    )

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
