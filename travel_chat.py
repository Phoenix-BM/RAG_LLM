import gradio as gr
from gradio.events import Events
import functools
import json
import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
_ = load_dotenv(find_dotenv())

from accurate_vec_retrieve import accurate_retrieve_vector_db

from accurate_llm_respond import llm_retrieved_respond

persist_directory = "xhs_data_base/xhs_vector_db/chroma"    #向量数据库的路径

def get_knowledges(query, top_k, chatbot, use_rag):
    if use_rag == '不使用RAG':
        return [' '] * 2 * 10
    top_k = int(top_k)

    history_query = ''
    if len(history_query) >= 1:
        history_query = ''.join([item[0] for item in chatbot if item is not None and item[0] is not None])

    sim_docs = accurate_retrieve_vector_db(query, history_query,use_rag, top_k)

    urls = []
    selected_knowledges = []

    for i, sim_doc in enumerate(sim_docs):
        # print(f"CD 检索到的第{i + 1}个内容: \n{sim_doc.page_content[:500]}", end="\n{'-'*30}\n")
        lines = sim_doc.page_content.splitlines()

        # 第2行以及第4行为换行符
        if len(lines) >= 3:
            title = lines[0].strip()
            url = lines[2].strip()
            content = lines[4].strip()

            urls.append(url)
            selected_knowledges.append(content)
    return selected_knowledges + urls


def respond(query, chat_history, topk_gr, use_rag=True, *selected_knowledge, respond_again=False):
    if not respond_again:
            if query is None or len(query.strip())==0:
                return "", chat_history

    else: # 撤销上一轮对话
        query = chat_history[-2][0]
        assert not query is None
        chat_history = chat_history[:-2]

    chat_history.append((query,None))

    topk_gr = int(topk_gr)
    bot_message = llm_retrieved_respond(query, chat_history, selected_knowledge, use_rag)

    chat_history.append((None, bot_message))
    
    return "", chat_history



if __name__ == "__main__":
    css = """
    #pdf {background-color: #F2FAFC}
    #chatbot {background-color: #BED0F9}
    #msg {background-color: #B3C4D4}
    footer {visibility: hidden}
    """

    with gr.Blocks(title='旅游攻略问答', css=css) as demo:
        with gr.Row():
            with gr.Column():
                topk_gr = gr.Slider(minimum=0, maximum=9, step=1, value=5, label="所需的知识的数量")
                
                # 检索出来的信息
                all_urls = []
                selected_knowledge = []

                with gr.Accordion('参考文档信息', open=False):
                    for topk_index in range(topk_gr.value):
                        with gr.Blocks():
                            text=gr.Textbox(placeholder=f'none', label=f'top-{topk_index + 1}', show_label=True, show_copy_button=True, lines=3)
                            with gr.Row():
                                url = gr.Textbox(placeholder=f'url', show_label=False)
                        selected_knowledge.append(text)
                        all_urls.append(url)

            with gr.Column():
                answerbot = gr.Markdown(label='参考答案',show_label=True, value = '', visible = True)
                chatbot = gr.Chatbot(label='对话窗口', show_label=True, bubble_full_width=False, elem_id="chatbot", \
                    elem_classes="feedback", height=500, avatar_images=['./imgs/user.png','./imgs/chatbot.png']) # 聊天框
                # 添加检索方式的单选按钮组
                use_rag = gr.Radio(
                        label="选择检索方式",
                        choices=["不使用RAG", "相似度检索", "最大边际相关性检索"],
                        value="相似度检索"
                    )
                msg = gr.Textbox(label='✏️ 用户输入', show_label=True, placeholder="", elem_id='msg') # 输入框
                    
                with gr.Row():
                    submit_again = gr.Button(value="🔄 重新生成回复", variant="secondary", interactive=False)
                    reset_button = gr.ClearButton([msg, chatbot, *selected_knowledge, *all_urls, answerbot], value="🧹 清空对话", variant="stop", interactive=False)
                    reset_button.click(fn=lambda : gr.update(value=3), outputs=[topk_gr])


        
        
        # 用户输入query并按下回车：检索+生成
        def disactive():
            return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        
        msg_click_event = msg.submit(fn=disactive, outputs=[msg, submit_again, reset_button]).then(
            fn = get_knowledges, \
            inputs=[msg, topk_gr, chatbot, use_rag], \
            outputs = selected_knowledge + all_urls).then(\
                    fn = functools.partial(respond, respond_again=False), \
                    inputs=[msg, chatbot, topk_gr, use_rag, *selected_knowledge], \
                    outputs=[msg, chatbot], queue=True).then(
                        fn=lambda : (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)), outputs=[msg, submit_again, reset_button])

        # 回滚
        submit_again_click_event = submit_again.click(fn=disactive, outputs=[msg, submit_again, reset_button]).then(
            fn = get_knowledges, \
            inputs=[msg, topk_gr, chatbot, use_rag], \
            outputs = selected_knowledge + all_urls).then(\
                    fn = functools.partial(respond, respond_again=True), \
                    inputs=[msg, chatbot, topk_gr, use_rag, *selected_knowledge], \
                    outputs=[msg, chatbot], queue=True).then(
                        fn=lambda : (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)), outputs=[msg, submit_again, reset_button])

        
    demo.queue().launch(server_port=8800, favicon_path='./imgs/logo.png', show_error=True, share=True)