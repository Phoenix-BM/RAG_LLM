import os

from zhipuai_llm import ZhipuAILLM

from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory

from langchain.chains import LLMChain

from creative_vec_retrieve import creative_retrieve_vector_db



database_path = "xhs_database/xhs_original_db"     # 源数据地址

persist_directory = "xhs_database/xhs_vector_db/chroma"    # 向量数据库地址



def llm_retrieved_respond(query):

    llm = ZhipuAILLM(model="glm-4", temperature=0.9, api_key=os.environ['ZHIPUAI_API_KEY'])

    template = """你现在是一个智能旅游问答助手。请使用以下的上下文来回答用户的问题。
                如果信息不在提供的上下文中，就说你不知道，
                就算你知道答案，如果上下文没有提供信息，就说不知道，
                不要借助外部知识回答问题，不要试图编造答案。
                如果你调用了上下文中的知识，请一定要提供对应的title和url。
                答案不要重复，尽量使答案详细具体，逻辑清晰。总是在回答的最后说“谢谢你的提问！”。
                {context}
                问题: {question}"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                    template=template)

    # 定义LLMChain
    qa_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 和prompt中的变量一致
        return_messages=True,       # 返回消息列表而不是单个字符串
    )

    chat_history = memory.load_memory_variables({})["chat_history"]
    chat_history_str = "\n\n".join([msg.content if hasattr(msg, 'content') and msg.content is not None else '' for msg in chat_history])

    # 将当前问题和会话历史融合
    full_query = chat_history_str + "\n\n" + query

    mmr_docs = creative_retrieve_vector_db(database_path, persist_directory, query, chat_history_str)

    # 将检索到的内容拼接成上下文
    context = "\n\n".join([doc.page_content for doc in mmr_docs])
    
    chat_history_messages = [msg.content if hasattr(msg, 'content') and msg.content is not None else '' for msg in chat_history]
    chat_history_str = "\n\n".join(chat_history_messages)

    full_context = chat_history_str + "\n\n" + context
 
    # 调用LLM生成答案
    response = qa_chain.run(context=full_context, question=query)
    print(response)

    # 更新会话历史
    memory.save_context({"question": query}, {"answer": response})

    