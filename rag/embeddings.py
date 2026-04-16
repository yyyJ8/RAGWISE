from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("QWEN_API_KEY")
)