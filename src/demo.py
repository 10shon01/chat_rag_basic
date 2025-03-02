import os
import re

import chainlit as cl
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyMuPDFLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader)
from langchain_community.vectorstores import FAISS
from openai import OpenAI

client = OpenAI()

settings = {
    "model": "gpt-4o",
    "temperature": 0.5,
}

file_content = ""
vector_store = None
last_user_message = ""


def vectorize_documents(documents):
    embeddings = OpenAIEmbeddings()
    texts = [doc.page_content for doc in documents]
    return FAISS.from_texts(texts, embeddings)


def search_documents(query, vector_store):
    if not isinstance(query, str):
        raise TypeError("クエリは文字列である必要があります。")
    try:
        results = vector_store.similarity_search_with_score(query)
    except Exception as e:
        raise ValueError(f"FAISS検索中にエラーが発生しました: {str(e)}")
    return [res[0].page_content for res in results[:3]]


def sanitize_text(text):
    text = re.sub(r'\b(role|assistant|system|user)\b\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<.*?>', '', text)
    return text


def create_prompt(user_message, relevant_texts):
    system_message = "あなたは優秀なQAシステムです。与えられた文書の内容に基づいて正確に回答してください。"
    assistant_message = "\n".join(relevant_texts)
    return [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": user_message.strip()},
    ]


async def call_openai_api(user_message, relevant_texts):
    try:
        messages = create_prompt(user_message, relevant_texts)
        response = client.chat.completions.create(
            model=settings['model'],
            messages=messages,
            temperature=settings['temperature'],
        )
        return response.choices[0].message.content, relevant_texts
    except Exception as e:
        return f"Error: {str(e)}", []


def format_sources(references):
    return "\n".join([f"参照元 {i+1}:\n{ref}" for i, ref in enumerate(references)])


async def send_feedback_options():
    feedback_options = [
        cl.Action(name="like", label="👍 良い", value='liked'),
        cl.Action(name="dislike", label="👎 悪い", value='disliked')
    ]
    await cl.Message(content="この回答は役に立ちましたか？", actions=feedback_options).send()


@cl.on_message
async def main(message: cl.Message):
    global file_content, vector_store, last_user_message

    user_message = message.content.strip()
    last_user_message = user_message

    if user_message.lower() == "ファイルアップロード":
        files = None
        while files is None:
            files = await cl.AskFileMessage(
                max_size_mb=20,
                content="PDF、Excel、またはWordファイルをアップロードしてください。",
                accept=[
                    "application/pdf",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ],
                raise_on_timeout=False,
            ).send()
        file = files[0]
        file_path = file.path
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif file_extension in [".xls", ".xlsx"]:
            loader = UnstructuredExcelLoader(file_path)
        elif file_extension in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            await cl.Message(content="対応していないファイル形式です。PDF、Excel、Wordをアップロードしてください。").send()
            return

        documents = loader.load()
        sanitized_docs = []
        sanitized_texts = []
        for doc in documents:
            clean_text = sanitize_text(doc.page_content)
            sanitized_texts.append(clean_text)
            sanitized_docs.append(type(doc)(page_content=clean_text, metadata=doc.metadata))

        vector_store = vectorize_documents(sanitized_docs)
        file_content = "\n".join(sanitized_texts)
        await cl.Message(content="ファイルが読み込まれました。質問を入力してください。").send()
    elif file_content:
        relevant_texts = search_documents(user_message, vector_store)
        tool_response, references = await call_openai_api(user_message, relevant_texts)
        await cl.Message(content=f"回答:\n{tool_response}").send()
        await cl.Message(content=f"以下の参照元を確認できます：\n{format_sources(references)}").send()
        await send_feedback_options()
    else:
        await cl.Message(content="先にファイルをアップロードしてください。").send()


@cl.action_callback("source_0")
async def show_source_0(action: cl.Action):
    await cl.Message(content=f"参照元 1 の内容:\n{action.value}").send()


@cl.action_callback("source_1")
async def show_source_1(action: cl.Action):
    await cl.Message(content=f"参照元 2 の内容:\n{action.value}").send()


@cl.action_callback("source_2")
async def show_source_2(action: cl.Action):
    await cl.Message(content=f"参照元 3 の内容:\n{action.value}").send()


@cl.action_callback("like")
async def like_feedback(action: cl.Action):
    await cl.Message(content="フィードバックをありがとう！").send()


@cl.action_callback("dislike")
async def dislike_feedback(action: cl.Action):
    global file_content, vector_store, last_user_message
    tool_response, references = await call_openai_api(last_user_message, [file_content])
    await cl.Message(content=f"再生成された回答:\n{tool_response}").send()
    await cl.Message(content=f"以下の参照元を確認できます：\n{format_sources(references)}").send()
    await send_feedback_options()
