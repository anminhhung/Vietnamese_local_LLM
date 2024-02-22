import pickle
from pathlib import Path
import streamlit as st 
import streamlit_authenticator as stauth 
import os
import logging
import requests
import ray
from src import utils

def send_query(text):
    resp = requests.post("http://localhost:8000/api/stream?query={}".format(text), stream=True)

    return resp

def run_app():    
    # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
    st.set_page_config(page_title="localbot", page_icon="🧑‍💼", layout="wide")

    st.title("💬 Chatbot")
    st.caption("🚀 I'm a Local Bot")
    
    # --- USER AUTHENTICATION ---
    names = ["hungam"]
    usernames = ["hungam"]

    # load hashed passwords
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)

    authenticator = stauth.Authenticate(names, usernames, hashed_passwords, 
                                        "sales_dashboard", "abcdef", cookie_expiry_days=30)

    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status == False:
        st.error("Username/password is incorrect")

    if authentication_status == None:
        st.warning("Please enter your username and password")

    if authentication_status:
        # ---- SIDEBAR ----
        authenticator.logout("Logout", "sidebar")
        st.sidebar.title(f"Welcome {name}")

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
                res = send_query(query)
                res.raise_for_status()
                # translate
                # res = translator.translate(res, dest="vi").text
                answer = res
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
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