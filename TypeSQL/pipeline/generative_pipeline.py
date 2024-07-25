import sys

from TypeSQL.logger import logging
from TypeSQL.exception import TypeSQLException

from dotenv import load_dotenv
from langchain_ai21 import AI21LLM

from TypeSQL.chain import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase

from TypeSQL.utils import TypeSQLCallbackHandler
from TypeSQL.prompt import STREAM_MSG_SCHEMA_MODE, STREAM_MSG

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

    @classmethod
    def change_query(cls, sql_dict, db) -> dict:
        logging.info("Inside change_query method of SQL2TextPipeline Class")
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
            logging.info("Exiting from change_query method of SQL2TextPipeline Class")
            return sql_dict

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e,sys)

    @classmethod
    def check_dml_ops(cls, sql_query: str) -> bool:
        """
        checking for DML operation performed by LLM so it doesn't need to perform write operations.
        :param sql_query:
        :return:
        """
        try:
            logging.info("Inside check_dml_ops method of SQL2TextPipeline Class")
            dml_ops_list = ['CREATE', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE', 'MERGE', 'UPSERT']

            if any(dml in sql_query for dml in dml_ops_list):
                logging.info(f'Query Contains the write operations: {sql_query}')
                logging.warning(f'Query Contains the write operations: {sql_query}')
                return False
            logging.info("Exiting check_dml_ops method of SQL2TextPipeline Class")
            return True

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e,sys)

    @classmethod
    def test_query(cls, db, sql_query: str) -> list:
        """
        Test whether SQL query is runnable.
        :param db:
        :param sql_query:
        :return:
        """
        try:
            logging.info("Inside test_query method of SQL2TextPipeline Class\n")
            return db.run(sql_query)
        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e, sys)

    def _get_db(self):
        """
        To obtain the created 'db' object
        :return:
        """
        try:
            return self.db
        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e, sys)

    def _get_response_prompt(self) -> str:
        """
        TO obtain the created object of prompts for human-readable response generations
        :return:
        """
        try:
            logging.info("Inside _get_response_prompt method of SQL2TextPipeline Class\n")
            return self.prompt_response
        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e, sys)

    def _init_llm(self, model_configs: dict):
        """
        Initialize the LLM model by connecting to AI21 Server.
        :param model_configs:
        :return:
        """
        try:
            logging.info("Inside _init_llm method of SQL2TextPipeline Class")
            self.llm_sql = AI21LLM(
                model=model_configs['model_sql'],
                base_url=model_configs['url_sql'] if model_configs['url_sql'] != '' else None,
                api_key=model_configs['apikey_sql'] if model_configs['apikey_sql'] != '' else None,
                temperature=model_configs['temperature_sql'],
                streaming=True,
                callbacks=[self.stream_handler_sql]
                )

            self.llm_sql_schema_mode = AI21LLM(
                model=model_configs['model_sql'],
                base_url=model_configs['url_sql'] if model_configs['url_sql'] != '' else None,
                api_key=model_configs['apikey_sql'] if model_configs['apikey_sql'] != '' else None,
                temperature=model_configs['temperature_sql'],
                streaming=True,
                callbacks=[self.stream_handler_sql_schema_model]
            )

            self.llm_response = AI21LLM(
                model=model_configs['model_response'],
                api_key=model_configs['apikey_response'] if model_configs['apikey_response'] != '' else None,
                base_url=model_configs['url_response'] if model_configs['url_response'] != '' else None,
                temperature=model_configs['temperature_response'],
                streaming=True,
                callbacks=[self.stream_handler_response]
            )
            logging.info("Inside _init_llm method of SQL2TextPipeline Class")

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e, sys)

    def _init_prompt(self, prompt_configs: dict) -> None:
        try:
            logging.info("Inside _init_prompt method of SQL2TextPipeline Class")
            prompt_validity, missing_keys, prompt = self._check_prompt_keys(prompt_configs)
            if prompt_validity:
                logging.info(f'Inside _init_prompt method of prompt validity of SQL2TextPipeline Class')
                self.prompt_sql = prompt_configs['prompt_sql']
                self.prompt_response = prompt_configs['prompt_response']
                self.prompt_regen_sql = prompt_configs['prompt_regen_sql']
                logging.info("Exiting _init_prompt method of SQL2TextPipeline Class")
            else:
                missing_keys = ', '.join(missing_keys) if isinstance(missing_keys, list) else missing_keys
                logging.info("Exiting _init_prompt method of SQL2TextPipeline Class")
                raise Exception(f'Missing keys {missing_keys} in {prompt}')
        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e, sys)

    def _check_prompt_keys(self, prompt_configs: dict):
        """
        check whether prompt keys are valid and required for inference.
        :param prompt_configs:
        :return:
        """
        try:
            keys_list = [
                ["{input}", "{table_info}", "{history}"],
                ["{history}","{question}", "{query}", "{results}"],
                ["{input}", "{table_info}", "{wrong_sql_query}"]
            ]

            prompt_list = ["prompt_sql", "prompt_response", "prompt_regen_sql"]
            display_name = ["prompt template of SQL Assistant",
                            "prompt template of AI Assistant",
                            "prompt template(Regeneration) for SQL Assistant"]

            for i, keys in enumerate(keys_list):
                missing_keys = [key for key in keys if key not in prompt_configs[prompt_list[i]]]
                if missing_keys != []:
                    return False, missing_keys, display_name[i]
            return True, None, None

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e, sys)

    def _init_db(self, configs) -> None:
        """
        Connect to database to get access to it.
        :param configs:
        :return:
        """
        try:
            logging.info("Inside _init_db method of SQL2TextPipeline Class")
            logging.info("Initializing database")

            if not configs['isdbSchemaMode']:
                if configs['db_type'] == 'SQLite':
                    address = f"sqlite:///sqlite/{configs['db_name_sqlite']}"
                elif configs['db_type'] == 'PostgreSQL':
                    address = f"postgrsql+psycopg2://{configs['username']}:{configs['password']}@{configs['hostname']}:{configs['port']}//{configs['db_name']}"
                elif configs['db_type'] == 'Mysql':
                    address = f"mysql+pymysql://{configs['username']}:{configs['password']}@{configs['hostname']}:{configs['port']}/{configs['db_name_mysql']}"
                self.db = SQLDatabase.from_uri(address, sample_rows_in_table_info=0)
            else:
                self.db = None

        except Exception as e:
            logging.error(e)
            raise TypeSQLException("Database connection error or database not found", sys)

    def create_sql_chain(self, mode: str, task: str):
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

            prompt = PromptTemplate.from_template(template = formatted_prompt_sql)
            if mode == 'standard':
                sql_chain = SQLDatabaseChain.from_llm(
                    self.llm_sql, self.db, prompt=prompt, return_sql=True, verbose=False, use_query_checker=False
                )
            elif mode == 'schema_mode':
                output_parser = StrOutputParser()
                sql_chain = prompt | self.llm_sql_schema_mode | output_parser

            return sql_chain, prompt

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e, sys)

    def create_response_chain(self):
        """
        Create the langchain object of response_chain for invoke in main application to create Human readable response based on SQL query.
        :param mode:
        :param task:
        :return:
        """
        try:
            output_parser = StrOutputParser()
            response_chain = self.llm_response | output_parser
            return response_chain

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e, sys)

    def setup(self, isSetup: bool, configs: dict, model_configs: dict, prompt_configs: dict) -> None:
        """
        Function to setup prompt the pipeline such prompt, llm and db connections.
        :param isSetup:
        :param configs:
        :param model_configs:
        :param prompt_configs:
        :return:
        """
        try:
            logging.info(f'Setup the pipeline {isSetup}')
            self._init_prompt(prompt_configs)
            self._init_llm(model_configs)
            self._init_db(configs)

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)





