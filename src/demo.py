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
document_metadata = []


def vectorize_documents(documents):
    embeddings = OpenAIEmbeddings()
    texts = [doc.page_content for doc in documents]
    global document_metadata
    document_metadata = [{"file_name": doc.metadata.get("source", "不明なファイル")} for doc in documents]
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store


def search_documents(user_query, vector_store):
    if not isinstance(user_query, str):
        raise TypeError("The user query must be a string.")
    try:
        results = vector_store.similarity_search_with_score(user_query)
    except Exception as e:
        raise ValueError(f"Error during FAISS search: {str(e)}")
    relevant_texts = [result[0].page_content for result in results[:3]]
    relevant_metadata = [result[0].metadata.get('source', '不明なファイル') for result in results[:3]]

    return relevant_texts, relevant_metadata


def sanitize_text(text):
    text = re.sub(r'\b(role|assistant|system|user)\b\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<.*?>', '', text)
    return text


def create_prompt(user_message, relevant_texts):
    system_message = "あなたは優秀なQAシステムです。与えられた文書の内容に基づいて正確に回答してください。"
    assistant_message = "\n".join(relevant_texts)
    user_message = user_message.strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": user_message},
    ]


async def tool(user_message, pdf_text, relevant_texts):
    """
    OpenAIのChatGPT APIにリクエストを送り、その応答を返す関数。
    """
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


user_message_global = ""


@cl.on_message
async def main(message: cl.Message):
    global file_content, vector_store, user_message_global

    user_message = message.content.strip().lower()
    user_message_global = user_message

    if user_message == "ファイルアップロード":
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

        sanitized_texts = [sanitize_text(doc.page_content) for doc in documents]
        sanitized_documents = [
            type(doc)(page_content=text, metadata=doc.metadata)
            for doc, text in zip(documents, sanitized_texts)
        ]
        vector_store = vectorize_documents(sanitized_documents)
        file_content = "\n".join(sanitized_texts)
        await cl.Message(content="ファイルが読み込まれました。質問を入力してください。").send()

    elif file_content:

        relevant_texts, relevant_metadata = search_documents(user_message, vector_store)

        tool_response, references = await tool(user_message, file_content, relevant_texts)

        await cl.Message(content=f"回答:\n{tool_response}").send()

        sources_content = "\n".join([f"参照元 {i+1}:\n{references[i]}" for i in range(len(references))])

        await cl.Message(content=f"以下の参照元を確認できます：\n{sources_content}").send()

        feedback_options = [
            cl.Action(name="like", label="👍 良い", value='liked'),
            cl.Action(name="dislike", label="👎 悪い", value='disliked')
        ]
        await cl.Message(content="この回答は役に立ちましたか？", actions=feedback_options).send()


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
    global file_content, vector_store, user_message_global

    tool_response, references = await tool(user_message_global, file_content, [file_content])

    await cl.Message(content=f"再生成された回答:\n{tool_response}").send()

    sources_content = "\n".join([f"参照元 {i+1}:\n{references[i]}" for i in range(len(references))])
    await cl.Message(content=f"以下の参照元を確認できます：\n{sources_content}").send()

    feedback_options = [
        cl.Action(name="like", label="👍 良い", value='liked'),
        cl.Action(name="dislike", label="👎 悪い", value='disliked')
    ]
    await cl.Message(content="この回答は役に立ちましたか？", actions=feedback_options).send()
