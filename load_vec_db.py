import json
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv, find_dotenv
from json_loader import UnstructuredJSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import math

database_path = "xhs_database/xhs_original_db"

persist_directory = "xhs_database/xhs_vector_db/chroma"

def load_vec_db(database_path, persist_directory):

    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()

    # 定义持久化路径,即是向量数据库路径
    persist_directory = "xhs_data_base/xhs_vector_db/chroma"

    # 加载环境变量
    _ = load_dotenv(find_dotenv())

    # 文件夹路径
    database_path = "xhs_data_base/xhs_original_db"

    # 遍历文件夹获取所有 JSON 文件路径
    file_paths = []
    for root, dirs, files in os.walk(database_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('.json'):
                file_paths.append(file_path)

    # 创建 UnstructuredJSONLoader 实例
    loaders = []
    for file_path in file_paths:
        loaders.append(UnstructuredJSONLoader(file_path))

    # 加载文档数据
    texts = []
    for loader in loaders:
        texts.extend(loader.load())

    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=100)

    # 分割文档
    split_docs = text_splitter.split_documents(texts)

    # 初始化 Chroma 向量数据库
    start = 0
    end = 0
    num_vec_store = math.ceil(len(split_docs) / 5) 

    while(end < len(split_docs)):
        start = end
        end += num_vec_store
        if(end > len(split_docs)):
            end = len(split_docs)

        print(f"正在将数据加载到向量数据库")

        vectordb = Chroma.from_documents(
            documents=split_docs[start : end], # 选择(end - start)个切分的 doc 进行加载
            embedding=embedding,
            persist_directory=persist_directory)
        
        print(f"向量库中存储的数量：{vectordb._collection.count()}")
        print(f"处理的文档数量：{len(split_docs)}") 
        progress = vectordb._collection.count() / len(split_docs) * 100
        print(f"加载进度：{progress:.2f}%")

    print("加载完毕")

load_vec_db(database_path, persist_directory)