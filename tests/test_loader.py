import os
import tempfile
from pathlib import Path
from rag.loader import DocumentLoader

def test_batch_upload():
    """测试批量上传10个不同格式文档"""
    test_docs = []
    
    # 创建临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建10个测试文件
        for i in range(1, 11):
            if i <= 3:
                # PDF文件
                file_path = os.path.join(tmpdir, f"test_{i}.pdf")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000102 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\n%%EOF")
            elif i <= 6:
                # Word文件
                file_path = os.path.join(tmpdir, f"test_{i}.docx")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"{{\"document\": \"Test document {i}\"}}")
            elif i <= 8:
                # Markdown文件
                file_path = os.path.join(tmpdir, f"test_{i}.md")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# Test Document {i}\n\nThis is a test markdown file.")
            else:
                # TXT文件
                file_path = os.path.join(tmpdir, f"test_{i}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"This is test text file {i}")
            
            test_docs.append(file_path)
        
        # 测试批量加载
        print("开始测试批量加载10个文档...")
        docs = DocumentLoader.load_documents(test_docs)
        print(f"成功加载 {len(docs)} 个文档")
        
        # 验证文档元数据
        for i, doc in enumerate(docs):
            print(f"\n文档 {i+1}:")
            print(f"  文件名: {doc.metadata.get('file_name', '未知')}")
            print(f"  来源: {doc.metadata.get('source', '未知')}")
            print(f"  文件大小: {doc.metadata.get('file_size', '未知')}")
            if doc.metadata.get('file_name', '').endswith('.pdf'):
                print(f"  页码: {doc.metadata.get('page', '未知')}")
            print(f"  上传时间: {doc.metadata.get('upload_time', '未知')}")

if __name__ == "__main__":
    test_batch_upload()