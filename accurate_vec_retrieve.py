'''
使用余弦相似度检索
只考虑检索出内容的相关性,但是可能会导致内容过于单一
'''

import json
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv, find_dotenv
from json_loader import UnstructuredJSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def accurate_retrieve_vector_db(query, history_query, use_rag, top_k=5):
    persist_directory = "xhs_database/xhs_vector_db/chroma"

    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()

    # 调用向量数据库
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory)

    full_query = history_query + "\n\n" + query
    if use_rag == '相似度检索':
        docs = vectordb.similarity_search(full_query, k=top_k)
    else:
        docs = vectordb.max_marginal_relevance_search(full_query, k=top_k)
    # print(f"检索到的内容数：{len(docs)}")

    return docs