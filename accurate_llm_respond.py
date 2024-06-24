import os

from zhipuai_llm import ZhipuAILLM

from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory

from langchain.chains import LLMChain

from accurate_vec_retrieve import accurate_retrieve_vector_db

database_path = "xhs_database/xhs_original_db"     # 源数据地址

persist_directory = "xhs_database/xhs_vector_db/chroma"    # 向量数据库地址


def llm_retrieved_respond(query, chat_history, knowledges, use_rag):
    llm = ZhipuAILLM(model="glm-4", temperature=0.9, api_key=os.environ['ZHIPUAI_API_KEY'])

    if use_rag == '不使用RAG':
        template = """你现在是一个智能旅游问答助手。请根据已有知识回答用户的问题。
                问题: {question}"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question"],
                                        template=template)
    else:
        template = """你现在是一个智能旅游问答助手。请使用以下的上下文来回答用户的问题。如果信息不在提供的上下文中，就说你不知道，
            就算你知道答案，如果上下文没有提供信息，就说不知道，不要借助外部知识回答问题，不要试图编造答案。
            如果回答的问题涉及到了景点，请根据现有的信息对景点作简单的介绍。
            在回答旅游攻略相关问题时，需要给出交通工具的选择建议，但是注意，你必须依据上下文来给出建议，上下文中没有提及的信息不要生成。
            在攻略最后需要给出一些注意事项，比如门票、天气等等，但是注意，如果给定的上下文没有对应的信息，就不要生成这一个注意事项（比如上下文中没有天气相关信息，就不要写天气相关的注意事项）。
            可以在合适的地方加一些emoji表情。
            尽量使答案详细具体。总是在回答的最后说“谢谢你的提问！”。
            上下文：{context}
            问题: {question}
            请检查自己的回答，有没有错误地将上下文中的信息当成问题的条件，如果有，请改正"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                        template=template)
        
    # 定义LLMChain
    qa_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)

    # 初始化一个空字符串用于存储对话历史
    dialogue_history_str = ""
    chat_history_str = ''

    # 遍历 chat_history 中的每个元组
    for query, response in chat_history:
        # 构造对话字符串，格式为 "用户提问：模型回答是。。。"
        dialogue_str = f"用户提问：{query} 模型回答是：{response}\n"
        # 将构造好的对话字符串添加到最终的字符串中
        dialogue_history_str += dialogue_str

    if use_rag:
        # 将检索到的内容拼接成上下文
        context = "\n\n".join([content for content in knowledges])
        
        chat_history_messages = [msg.content if hasattr(msg, 'content') and msg.content is not None else '' for msg in chat_history]
        chat_history_str = "\n\n".join(chat_history_messages)

        full_context = chat_history_str + "\n\n" + context
    else:
        chat_history_messages = [msg.content if hasattr(msg, 'content') and msg.content is not None else '' for msg in chat_history]
        chat_history_str = "\n\n".join(chat_history_messages)

        full_context = chat_history_str

 
    # 调用LLM生成答案
    response = qa_chain.run(context=full_context, question=query)
    return response