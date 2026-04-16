import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

def test_health_check():
    """测试健康检查接口"""
    response = requests.get(f"{API_URL}/health")
    print("健康检查:", response.status_code, response.json())
    assert response.status_code == 200

def test_upload():
    """测试文件上传接口"""
    files = [
        ("files", ("test.txt", b"This is a test file"))
    ]
    data = {"api_key": API_KEY}
    response = requests.post(f"{API_URL}/upload", files=files, data=data)
    print("文件上传:", response.status_code, response.json())
    assert response.status_code == 200

def test_query():
    """测试查询接口"""
    data = {
        "questions": "RAGWise支持哪些文档格式？",
        "api_key": API_KEY
    }
    response = requests.post(f"{API_URL}/query", data=data)
    print("查询:", response.status_code, response.json())
    assert response.status_code == 200

if __name__ == "__main__":
    print("测试后端API...")
    test_health_check()
    test_upload()
    test_query()
    print("所有测试通过！")