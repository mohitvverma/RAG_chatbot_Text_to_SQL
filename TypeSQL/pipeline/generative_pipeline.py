from transformers import Pipeline

from TypeSQL.logger import logging
from TypeSQL.exception import TypeSQLException

from dotenv import load_dotenv
from langchain_ai21 import AI21LLM
from langchain.prompts import Prompts, PromptTemplate
from langchain_openai import OpenAI

from langchain.chains import Chains
from langchain_community

from TypeSQL.chain import SQLDatabaseChain

from TypeSQL.utils import TypeSQLCallbackHandler
from TypeSQL.prompt import SCHEMA_PROMPT, STREAM_MSG_SCHEMA_MODE, STREAM_MSG

load_dotenv()
logging.info("Environment Variables Loaded")

class SQL2TextPipeline():
    """
    A SQL to Text Pipeline that contains the setup of two LLMs, which are sqlcoder-7b-2 and Mistral-7B-Instruct-v0.3.

    sqlcoder-7b02 used is a quantized 8-bit precision gguf model to generate Text-to-SQL conversion.
    It is a finetuned model of CodeLlama-7B to be specialised at Natural Language to SQL generation.

    Mistral-7B-Instruct-v0.3 used is a quantized 8-bit precision gguf model to generate human response based on sql results.
    Both of the models are served by llama.cpp server and it has the same OpenAI API policy, thus it serves as a good
    alternative as OpenAI-compatible server.
    The pipeline also contains setup of SQLite database. In the end, the pipeline will return two langchains to the main
    application to do inference by invoking the chains.

    Models used in this application:
    Model_sql     : https://huggingface.co/MaziyarPanahi/sqlcoder-7b-2-GGUF
    Model_response: https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF
    """

    def __init__(self):
        self.stream_handler_response = TypeSQLCallbackHandler()
        self.stream_handler_sql = TypeSQLCallbackHandler(initial_text=STREAM_MSG)
        self.stream_handler_sql_schema_model = TypeSQLCallbackHandler(initial_text=STREAM_MSG_SCHEMA_MODE)

    @staticmethod
    def change_query(cls, sql_dict, db):
        """
        Change query if DB is sqlite3
        :param cls:
        :param sql_dict:
        :param db:
        :return:
        """
        try:
            if db.dialect != 'sqlite':
                return sql_dict

            if "ilike" in sql_dict['result'].lower():
                sql_dict['result'] = sql_dict['result'].replace('ilike', 'like')
                sql_dict['result'] = sql_dict['result'].replace('ILIKE', 'LIKE')

            return sql_dict

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    @classmethod
    def check_dml_ops(cls, sql_query: str) -> bool:
        """
        checking for DML operation performed by LLM so it doesn't need to perform write operations.
        :param sql_query:
        :return:
        """
        try:
            dml_ops_list = ['CREATE', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE', 'MERGE', 'UPSERT']

            if any(dml in sql_query for dml in dml_ops_list):
                logging.info(f'Query Contains the write operations: {sql_query}')
                return False
            return True

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    @classmethod
    def test_query(cls,db, sql_query: str) -> bool:
        """
        Test whether SQL query is runnable.
        :param db:
        :param sql_query:
        :return:
        """
        try:
            db.run(sql_query)
            return True

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    def _get_db(self):
        """
        To obtain the created 'db' object
        :return:
        """
        try:
            return self.db

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    def _get_response_prompt(self):
        """
        TO obtain the created object of prompts for human-readable response generations
        :return:
        """
        try:
            return self.prompt_response
        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    def _init_llm(self, model_configs: dict):
        """
        Initialize the LLM model by connecting to AI21 Server.
        :param model_configs:
        :return:
        """
        try:
            self.llm_sql = AI21LLM(
                model = model_configs['model_sql'],
                base_url = model_configs['base_url'],
                api_key = model_configs['api_key'],
                temperature = model_configs['temperature'],
                streaming = True,
                callbacks = [self.stream_handler_sql]
                )

            self.llm_sql_schema_mode = AI21LLM(
                model = model_configs['model_sql'],
                base_url = model_configs['base_url'],
                api_key = model_configs['api_key'],
                temperature = model_configs['temperature'],
                streaming = True,
                callbacks = [self.stream_handler_sql_schema_model]
            )

            self.llm_response = AI21LLM(
                model = model_configs['model_sql'],
                api_key = model_configs['api_key'],
                base_url = model_configs['base_url'],
                temperature = model_configs['temperature'],
                streaming = True,
                callbacks = [self.stream_handler_response]
            )

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    def __init_prompt(self, prompt_configs: dict) -> None:
        try:
            logging.info("Initializing Prompt Configs")
            prompt_validity, missing_keys, prompt = self._check_prompt_keys(prompt_configs)

            if prompt_validity:
                self.prompt_sql = prompt_configs['prompt_sql']
                self.prompt_response = prompt_configs['prompt_reponse']
                self.prompt_regen_sql = prompt_configs['prompt_regen_sql']
            else:
                missing_keys = ', '.join(missing_keys) if isinstance(missing_keys, list) else missing_keys
                raise TypeSQLException(f'Missing keys {missing_keys} in {prompt}')


    def _check_prompt_keys(self, prompt_configs: dict) -> bool:
        """
        check whether prompt keys are valid and required for inference.
        :param prompt_configs:
        :return:
        """
        try:
            keys_list = [
                ["{input}", "{table_info}", "{history}"],
                ["{history}","{question}", "{query}", "{results}"],
                [f"{input}", "{table_info}", "{wrong_sql_query}"]
            ]

            prompt_list = ["prompt_sql", "prompt_reponse", "prompt_regen_sql"]
            display_name = ["prompt template for SQL Assistant",
                            "prompt template for AI Assistant",
                            "prompt template(Regeneration) for SQL Assistant"]

            for i, keys in enumerate(keys_list):
                missing_keys = [key for key in keys if key not in prompt_configs[prompt_list[i]] ]
                if missing_keys != []:
                    return False missing_keys, display_name[i]

            return True, None, None

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    def _init_db(self, configs) -> None:
        """
        Connect to database to get access to it.
        :param configs:
        :return:
        """
        try:
            logging.info("Initializing DB")
            if not configs['isdbSchemaMode']:
                if configs['db_type'] == 'sqlite':
                    address = f"sqlite:///{configs['db_name_sqlite']}"
                elif configs['db_type'] == 'PostgreSQL':
                    address = f"postgrsql+psycopg2://{configs['username']}:{configs['password']}@{configs['hostname']}:{configs['port']}//{configs['db_name']}"
                elif configs['db_type'] == 'Mysql':
                    address = f"mysql+pymysql://{configs['username']}:{configs['password']}@{configs['hostname']}:{configs['port']}/{configs['db_name_mysql']}"

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    def create_sql_chain(self, mode: str, task:str) -> None:
        """
        Create Langchain object of SQL Chain to invoke in main applications to create SQL query based on user input
        :param mode:
        :param task:
        :return:
        """
        try:
            if task == 'gen':
                formatted_prompt_sql = self.prompt_sql

            elif task == 'regen':
                formatted_prompt_sql = self.prompt_regen_sql

            prompt = PromptTemplate.from_template(template=formatted_prompt_sql)
            if mode == 'standard':
                sql_chain = SQLDatabaseChain()





    def setup(self, isSetup: bool, configs: dict, model_configs: dict, prompt_configs: dict) -> None:
        """
        Function to setup the pipeline such prompt, llm and db connections.
        :param isSetup:
        :param configs:
        :param model_configs:
        :param prompt_configs:
        :return:
        """
        try:
            logging.info(f'Setup the pipeline {isSetup}')





