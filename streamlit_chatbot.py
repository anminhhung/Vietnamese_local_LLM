import os
import logging
import requests
import ray
import streamlit as st
import src.utils as utils

@ray.remote
def send_query(text):
    resp = requests.get("http://localhost:8000/?query={}".format(text))

    return resp.text


def run_app():    
    st.title("💬 Chatbot")
    st.caption("🚀 I'm a Local Bot")
    
    # show source use to debug
    # show_sources = st.checkbox("Show sources", value=False)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì được cho bạn?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
        
    # Initialize the QA system using caching
    # translater = Translation(from_lang="en", to_lang='vi', mode='translate') 

    if query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)
        
        # Add spinner
        with st.spinner("Thinking..."):
            res = ray.get(send_query.remote(query))
            answer, docs = res["result"], res["source_documents"]

        # if translate_output:
        #     answer = translater(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        # st.chat_message("assistant").write(answer)
        
        response = answer
        
        # if show_sources:
        #     response += "\n\n"
        #     response += "----------------------------------SOURCE DOCUMENTS---------------------------\n"
        #     for document in docs:
        #         response += "\n> " + document.metadata["source"] + ":\n" + document.page_content
        #     response += "----------------------------------SOURCE DOCUMENTS---------------------------\n"
        
        # save_qa
        utils.log_to_csv(query, answer)
            
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    run_app()