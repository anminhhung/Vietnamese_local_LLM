import pickle
from pathlib import Path
import streamlit as st 
import os
import logging
import requests
import ray
from src import utils
from io import StringIO

def send_query(text, file=None):

    if file:
        file_name = file.name
        file = file.read()
        file = str(file,'utf-8')
        file = StringIO(file)        
        data = {"query": text}
        files = {'file': (file_name, file, 'text/csv')}

        resp = requests.post("http://localhost:8000/api/data-analytic/", files=files, data=data, stream=False)
    else:
        resp = requests.post("http://localhost:8000/api/stream?query={}".format(text), stream=True)

    return resp

def run_app():    
    # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
    st.set_page_config(page_title="localbot", page_icon="🧑‍💼", layout="wide")

    st.title("💬 Chatbot")
    st.caption("🚀 I'm a Local Bot")
    uploaded_file = st.file_uploader("Chosose a CSV file", type=["csv"])

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
            if uploaded_file is not None:
                res = send_query(query, uploaded_file)
            else:
                res = send_query(query)

            res.raise_for_status()
            # translate
            # res = translator.translate(res, dest="vi").text
            answer = res
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

            if uploaded_file is None: 
                for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            else:
                full_response = res.text
            
            message_placeholder.markdown(full_response)

        # if translate_output:
        #     answer = translater(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # st.chat_message("assistant").write(answer)
        
        # response = answer
        
        # if show_sources:
        #     response += "\n\n"
        #     response += "----------------------------------SOURCE DOCUMENTS---------------------------\n"
        #     for document in docs:
        #         response += "\n> " + document.metadata["source"] + ":\n" + document.page_content
        #     response += "----------------------------------SOURCE DOCUMENTS---------------------------\n"
        
        # save_qa
        utils.log_to_csv(query, answer)
        st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    run_app()