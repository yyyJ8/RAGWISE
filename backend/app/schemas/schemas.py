from pydantic import BaseModel,Field
from typing import Optional,List

class UploadRequest(BaseModel):
    pass

class UploadResponse(BaseModel):
    status:str
    message:str
    file_count:int
    chunk_count:int

class QueryRequest(BaseModel):
    question:str=Field(...,description="查询问题")

class QueryResponse(BaseModel):
    status:str
    answer:str
    sources: Optional[List[str]] = None

class ChatRequest(BaseModel):
    question:str=Field(...,description="聊天问题")

class HealthCheckResponse(BaseModel):
    status:str
    service:str

class ErrorResponse(BaseModel):
    status:str
    error:str
    message:Optional[str]=None

class APIKeyRequest(BaseModel):
    api_key:str=Field(...,description="API密钥")


