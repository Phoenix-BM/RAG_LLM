'''
使用最大边际相关性 (MMR, Maximum marginal relevance) 检索
可以帮助我们在保持相关性的同时，增加内容的丰富度
核心思想是在已经选择了一个相关性高的文档之后，再选择一个与已选文档相关性较低但是信息丰富的文档。
这样可以在保持相关性的同时，增加内容的多样性，避免过于单一的结果
'''

import json
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv, find_dotenv
from json_loader import UnstructuredJSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

persist_directory = "xhs_database/xhs_vector_db/chroma"

def accurate_retrieve_vector_db(persist_directory, query, history_query):

    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()

    # 调用向量数据库
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory)

    full_query = history_query + "\n\n" + query

    mmr_docs = vectordb.max_marginal_relevance_search(full_query, k=5)
    print(f"检索到的内容数：{len(mmr_docs)}")

    # urls = []
    # selected_knowledges = []

    # for i, sim_doc in enumerate(mmr_docs):
    #     # print(f"CD 检索到的第{i + 1}个内容: \n{sim_doc.page_content[:500]}", end="\n{'-'*30}\n")
    #     lines = sim_doc.page_content.splitlines()

    #     # 第2行以及第4行为换行符
    #     if len(lines) >= 3:
    #         title = lines[0].strip()
    #         url = lines[2].strip()
    #         content = lines[4].strip()

    #         urls.append(url)
    #         selected_knowledges.append(content)

    #         print(f"CD检索到的第{i + 1}个内容: \n")
    #         print(f"Title: {title}")
    #         print(f"URL: {url}")
    #         print(f"Content: {content}")
    #     else:
    #         print("Invalid page_content format.")

    return mmr_docs