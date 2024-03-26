import sys
import os
from llama_index.core import ChatPromptTemplate, PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.constants import SYSTEM_PROMPT, USER_PROMPT, REFINE_PROMPT



def get_prompt_template(system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT):
    message_template = [
        ChatMessage(content=system_prompt, role=MessageRole.SYSTEM),
        ChatMessage(content=user_prompt, role=MessageRole.USER)
    ]
    chat_template = ChatPromptTemplate(message_templates=message_template)
    refine_template = PromptTemplate(REFINE_PROMPT)
    
    return chat_template, refine_template