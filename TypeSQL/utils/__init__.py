from typing import Any
import streamlit as st
import sys

from TypeSQL.logger import logging
from TypeSQL.exception import TypeSQLException

from langchain.callbacks.base import BaseCallbackHandler


class TypeSQLCallbackHandler(BaseCallbackHandler):
    """
    Base callback handler that can be used to handle callbacks from langchain.

    Callbacks : In the context of machine learning, callbacks are often used to execute code at certain stages of the training process,
    such as at the end of an epoch, batch, or training session. They are commonly used for tasks like logging, adjusting learning rates,
    saving models, or implementing early stopping.

    Use case : To stream generated token in main application and print as 'assistant' message or 'sql' message
    """
    def __init__(self, initial_text =None):
        logging.info('Initializing TypeSQLCallbackHandler')
        self.initial_text = initial_text
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> Any:
        try:
            """
            Stream tokens when producing new token.
    
            :param token:
            :param kwargs:
            :return: Return the new token, when generated.
            """
            self.text += token
            st.markdown(self.text, unsafe_allow_html=True)
            logging.info(f'Inside and Exiting from the TypeSQLCallbackHandler on_llm_new_token')
        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    def on_llm_end(self, response, **kwargs):
        try:
            """
            Print the whole message when LLM stop generating.
            :param response:
            :param kwargs:
            :return: None
            """
            if self.initial_text != "":
                st.markdown(f"{self.text} \n ``")
                logging.info(f'Inside and Exiting from the TypeSQLCallbackHandler on_llm_end')

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e) from e

