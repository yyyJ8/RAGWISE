import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

def test_frontend_backend_integration():
    """测试前后端集成"""
    print("测试前后端集成...")
    
    # 测试健康检查
    response = requests.get(f"{API_URL}/health")
    print(f"健康检查: {response.status_code} - {response.json()}")
    
    # 测试文件上传
    files = [
        ("files", ("test.txt", b"This is a test file for frontend integration"))
    ]
    data = {"api_key": API_KEY}
    response = requests.post(f"{API_URL}/upload", files=files, data=data)
    print(f"文件上传: {response.status_code} - {response.json()}")
    
    # 测试查询
    data = {
        "question": "RAGWise支持哪些文档格式？",
        "api_key": API_KEY
    }
    response = requests.post(f"{API_URL}/query", data=data)
    print(f"查询: {response.status_code} - {response.json()}")
    
    print("前后端集成测试完成！")

if __name__ == "__main__":
    test_frontend_backend_integration()