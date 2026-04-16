import gradio as gr
import requests
import os
import time
from dotenv import load_dotenv
import mimetypes
load_dotenv()

API_URL = os.getenv("API_URL","http://localhost:8000")
API_KEY = os.getenv("API_KEY")

def upload_files(files):
    try:
        #files_data=[("files",(file.name,open(file.name,"rb"),file.type)) for file in files]
        files_data=[]
        for file in files:
            mime_type,_=mimetypes.guess_type(file.name)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            files_data.append(("files",(file.name,open(file.name,"rb"),mime_type)))
        data={"api_key": API_KEY}
        response=requests.post(
            f"{API_URL}/upload",
            files=files_data,
            data=data
        )
        response.raise_for_status()
        result = response.json()
        if result["status"] == "success":
            return f"成功上传 {result['file_count']} 个文件，生成 {result['chunk_count']} 个文档块"
        else:
            return f"上传失败：{result['message']}"
    except Exception as e:
        return f"上传过程中出错：{str(e)}"

def query_rag(question):
    try:
        response = requests.post(
            f"{API_URL}/query",
            data={
                "questions": question,
                "api_key": API_KEY
            }
        )
        response.raise_for_status()
        result = response.json()
        if result["status"] == "success":
            answer = result["answer"]
            sources = result.get("sources", [])
            if sources:
                answer += "\n\n**来源：**\n"
                for i, source in enumerate(sources, 1):
                    answer += f"{i}. {source}\n"
            return answer
        else:
            return f"查询失败：{result['message']}"
    except Exception as e:
        return {f"查询过程中出错：{str(e)}"}

def chat_rag(questions, history):
    try:
        # 添加用户消息到历史记录
        history.append({"role": "user", "content": questions})
        
        response = requests.post(
            f"{API_URL}/chat",
            data={
                "questions": questions,
                "api_key": API_KEY
            },
            stream=True
        )
        response.raise_for_status()
        answer = ""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                answer += chunk.decode('utf-8')
                
                temp_history = history.copy()
                
                if temp_history and temp_history[-1].get("role") == "assistant":
                    temp_history.pop()
                
                temp_history.append({"role": "assistant", "content": answer})
                yield temp_history,""
    except Exception as e:
        # 出错时返回错误消息作为助手消息
        error_message = f"聊天过程中出错：{str(e)}"
        history.append({"role": "assistant", "content": error_message})
        yield history,""

with gr.Blocks(
    title="RAGWise 智询库",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("""
    # RAGWise 智询库
    智能文档知识库系统
    """)
    with gr.Tab("文件上传"):
        file_upload = gr.File(
            file_count="multiple",
            file_types=[".pdf", ".docx", ".md", ".txt"],
            label="选择文件"
        )
        upload_button = gr.Button("上传并处理")
        upload_output = gr.Textbox(
            label="上传结果",
            lines=3,
            interactive=False
        )
        upload_button.click(
            fn=upload_files,
            inputs=file_upload,
            outputs=upload_output
        )

    with gr.Tab("智能问答"):
        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入您的问题...",
                    lines=2
                )
                query_button = gr.Button("查询", variant="primary")
            with gr.Column(scale=1):
                api_key_input = gr.Textbox(
                    label="API Key",
                    value=API_KEY,
                    type="password"
                )
        # answer_output = gr.Textbox(
        #     label="回答",
        #     lines=10,
        #     interactive=False
        # )    
        
        # sources_output = gr.Textbox(
        #     label="来源信息",
        #     lines=3,
        #     interactive=False
        # )
        answer_output = gr.Markdown()
        sources_output = gr.Markdown()
        def query_with_sources(question):
            answer = query_rag(question) 
            if "**来源：**" in answer:
                answer_part, sources_part = answer.split("**来源：**", 1)
                return answer_part.strip(), sources_part.strip()
            return answer, ""

        query_button.click(
        fn=query_with_sources,
        inputs=question_input,
        outputs=[answer_output, sources_output]
        ) 

    with gr.Tab("聊天模式"):
        chatbot = gr.Chatbot()
        chat_input = gr.Textbox(
            label="输入消息",
            placeholder="请输入您的问题...",
            lines=2
        )
        chat_submit = gr.Button("发送", variant="primary")
        chat_submit.click(
            fn=chat_rag,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )
        chat_input.submit(
            fn=chat_rag,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input]
        )

    gr.Markdown("""
    ---
    **使用说明：**
    1. 在「文件上传」标签页上传PDF、Word、Markdown或TXT文件
    2. 在「智能问答」标签页输入问题，获取带来源引用的回答
    3. 在「聊天模式」标签页进行流式对话
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )