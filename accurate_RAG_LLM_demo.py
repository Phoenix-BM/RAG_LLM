import json
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv, find_dotenv
from json_loader import UnstructuredJSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from zhipuai_llm import ZhipuAILLM

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationalRetrievalChain

from langchain.chains import LLMChain


# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# 定义持久化路径
persist_directory = "xhs_database/xhs_vector_db/chroma"

# 加载环境变量
_ = load_dotenv(find_dotenv())

# 文件夹路径
folder_path = "xhs_database/xhs_original_db"

# 遍历文件夹获取所有 JSON 文件路径
file_paths = []
for root, dirs, files in os.walk(folder_path):
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

# 初始化并加载 Chroma 向量数据库
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory)

# 输出向量库中存储的数量和处理的文档数量
print(f"向量库中存储的数量：{vectordb._collection.count()}")
print(f"处理的文档数量：{len(split_docs)}")

llm = ZhipuAILLM(model="glm-4", temperature=0.9, api_key=os.environ['ZHIPUAI_API_KEY'])

template = """你现在是一个智能旅游问答助手。请使用以下的上下文来回答用户的问题。如果信息不在提供的上下文中，就说你不知道，
            就算你知道答案，如果上下文没有提供信息，就说不知道，不要借助外部知识回答问题，不要试图编造答案。
            如果回答的问题涉及到了景点，请根据现有的信息对景点作简单的介绍。
            在回答旅游攻略相关问题时，需要给出交通工具的选择建议，但是注意，你必须依据上下文来给出建议，上下文中没有提及的信息不要生成。
            在攻略最后需要给出一些注意事项，比如门票、天气等等，但是注意，如果给定的上下文没有对应的信息，就不要生成这一个注意事项（比如上下文中没有天气相关信息，就不要写天气相关的注意事项）。
            可以在合适的地方加一些emoji表情。
            尽量使答案详细具体。总是在回答的最后说“谢谢你的提问！”。
            上下文：{context}
            问题: {question}"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                 template=template)

# 定义LLMChain
qa_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

# 初始化会话内存
memory = ConversationBufferMemory(
    memory_key="chat_history",  # 和prompt中的变量一致
    return_messages=True,       # 返回消息列表而不是单个字符串
)

while True:
    query = input("请输入问题（输入'退出'结束程序）：")

    if query == "退出":
        break

    # 获取会话历史并将其转换为字符串列表

    chat_history = memory.load_memory_variables({})["chat_history"]
    chat_history_str = "\n\n".join([msg.content if hasattr(msg, 'content') and msg.content is not None else '' for msg in chat_history])

    # 将当前问题和会话历史融合
    full_query = chat_history_str + "\n\n" + query

    # 使用余弦相似度检索
    sim_docs = vectordb.similarity_search(full_query, k=5)
    print(f"检索到的内容数：{len(sim_docs)}")

    for i, sim_doc in enumerate(sim_docs):
        # print(f"CD 检索到的第{i + 1}个内容: \n{sim_doc.page_content[:500]}", end="\n{'-'*30}\n")
        lines = sim_doc.page_content.splitlines()

        # 第2行以及第4行为换行符
        if len(lines) >= 3:
            title = lines[0].strip()
            url = lines[2].strip()
            content = lines[4].strip()

            print(f"CD检索到的第{i + 1}个内容: \n")
            print(f"Title: {title}\n")
            print(f"URL: {url}\n")
            print(f"Content: {content}\n")
        else:
            print("Invalid page_content format.")

    # 将检索到的内容拼接成上下文
    context = "\n\n".join([doc.page_content for doc in sim_docs])
    
    chat_history_messages = [msg.content if hasattr(msg, 'content') and msg.content is not None else '' for msg in chat_history]
    chat_history_str = "\n\n".join(chat_history_messages)

    full_context = chat_history_str + "\n\n" + context
 
    # 调用LLM生成答案
    response = qa_chain.run(context=full_context, question=query)
    print(response)

    # 更新会话历史
    memory.save_context({"question": query}, {"answer": response})

