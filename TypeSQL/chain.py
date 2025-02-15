"""
Custom SQL Database chain
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from typing import Any, List, Optional, Dict
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.schema import BasePromptTemplate
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain.prompts import PromptTemplate

from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models import BaseLanguageModel

from langchain_experimental.pydantic_v1 import Extra, Field

from TypeSQL.logger import logging
from TypeSQL.exception import TypeSQLException

INTERMEDIATE_STEPS_KEY = "intermediate_steps"
SQL_QUERY = "SQLQuery:"
SQL_RESULT = "SQLResult:"


class SQLDatabaseChain(BaseLanguageModel):
    logging.info("Initializing SQLDatabaseChain")
    """
    Chain for interacting with SQL Database.

    Example:
            code-block:: python

            from langchain_experimental.sql import SQLDatabaseChain
            from langchain_community.llms import OpenAI, SQLDatabase
            db = SQLDatabase(...)
            db_chain = SQLDatabaseChain.from_llm(OpenAI(), db)

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include the permissions this chain needs.
        Failure to do so may result in data corruption or loss, since this chain may
        attempt commands like `DROP TABLE` or `INSERT` if appropriately prompted.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this chain.
        This issue shows an example negative outcome if these steps are not taken:
        https://github.com/langchain-ai/langchain/issues/5923
    """

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None

    database: SQLDatabase = Field(exclude=True)

    """
    SQL Database chain to connect
    """
    prompt: Optional[PromptTemplate] = None

    """
    [] Prompt template to use to translate natural query to SQL
    """

    top_k: int = 5
    """ Number of result to return from any query"""

    input_key: str = 'query'

    output_key: str = 'result'

    return_sql: bool = False

    return_intermediate_steps: bool = False

    return_direct: bool = False

    use_query_checker: bool = False

    query_checker_prompt: Optional[PromptTemplate] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    # @root_validator(pre=True)
    # def raise_deprecation(cls, values: Dict) -> Dict:
    #     if "llm" in values:
    #         warnings.warn(
    #             "Directly instantiating an SQLDatabaseChain with an llm is deprecated. "
    #             "Please instantiate with llm_chain argument or using the from_llm "
    #             "class method."
    #         )
    #         if "llm_chain" not in values and values["llm"] is not None:
    #             database = values["database"]
    #             prompt = values.get("prompt") or SQL_PROMPTS.get(
    #                 database.dialect, PROMPT
    #             )
    #             values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
    #     return values

    @property
    def input_keys(self) -> List[str]:
        """
        Return the singular input key.
        meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        meta private:
        """
        try:
            if not self.return_intermediate_steps:
                return [self.output_key]
            else:
                return [self.output_key, INTERMEDIATE_STEPS_KEY]

        except Exception as e:
            logging.error(e)
            raise TypeSQLException(e)

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_text = f"{inputs[self.input_key]}"
        extra = inputs["extra"]  # modified parts
        _run_manager.on_text(input_text, verbose=self.verbose)

        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        llm_inputs = {
            "input": input_text,
            "top_k": str(self.top_k),
            "history": str(extra),  # modified parts
            "wrong_sql_query": str(extra),  # modified parts
            "dialect": self.database.dialect,
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }

        prompt = self.prompt.format(**llm_inputs)
        if self.memory is not None:
            for k in self.memory.memory_variables:
                llm_inputs[k] = inputs[k]

        intermediate_steps: List = []
        try:
            intermediate_steps.append(llm_inputs.copy())  # input: sql generation
            sql_cmd = self.llm_chain.predict( callbacks=_run_manager.get_child(), **llm_inputs).strip()
            # if "ilike" in sql_cmd.lower():
            #     sql_cmd = sql_cmd.replace("ilike","like")
            #     sql_cmd = sql_cmd.replace("ILIKE","LIKE")

            if self.return_sql:
                return {self.output_key: sql_cmd, "prompt": prompt}
            if not self.use_query_checker:
                _run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
                intermediate_steps.append(
                    sql_cmd
                )  # output: sql generation (no checker)
                intermediate_steps.append({"sql_cmd": sql_cmd})  # input: sql exec
                # if SQL_QUERY in sql_cmd:
                #     sql_cmd = sql_cmd.split(SQL_QUERY)[1].strip()
                # if SQL_RESULT in sql_cmd:
                #     sql_cmd = sql_cmd.split(SQL_RESULT)[0].strip()
                result = self.database.run(sql_cmd)
                intermediate_steps.append(str(result))  # output: sql exec

            else:
                query_checker_prompt = self.query_checker_prompt or PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["query", "dialect"]
                )
                query_checker_chain = LLMChain(
                    llm=self.llm_chain.llm, prompt=query_checker_prompt
                )
                query_checker_inputs = {
                    "query": sql_cmd,
                    "dialect": self.database.dialect,
                }

                checked_sql_command: str = query_checker_chain.predict(
                    callbacks=_run_manager.get_child(), **query_checker_inputs
                ).strip()
                intermediate_steps.append(
                    checked_sql_command
                )  # output: sql generation (checker)
                _run_manager.on_text(
                    checked_sql_command, color="green", verbose=self.verbose
                )
                intermediate_steps.append(
                    {"sql_cmd": checked_sql_command}
                )  # input: sql exec
                result = self.database.run(checked_sql_command)
                intermediate_steps.append(str(result))  # output: sql exec
                sql_cmd = checked_sql_command

            _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
            _run_manager.on_text(str(result), color="yellow", verbose=self.verbose)

            # If return direct, we just set the final result equal to
            # the result of the sql query result, otherwise try to get a human readable
            # final answer
            if self.return_direct:
                final_result = result
            else:
                _run_manager.on_text("\nAnswer:", verbose=self.verbose)
                input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
                llm_inputs["input"] = input_text
                intermediate_steps.append(llm_inputs.copy())  # input: final answer
                final_result = self.llm_chain.predict(
                    callbacks=_run_manager.get_child(),
                    **llm_inputs,
                ).strip()
                intermediate_steps.append(final_result)  # output: final answer
                _run_manager.on_text(final_result, color="green", verbose=self.verbose)
            chain_result: Dict[str, Any] = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
            return chain_result

        except Exception as exc:
            # Append intermediate steps to exception, to aid in logging and later
            # improvement of few shot prompt seeds
            exc.intermediate_steps = intermediate_steps  # type: ignore
            logging.error(exc)
            raise TypeSQLException(exc)

    @property
    def _chain_type(self) -> str:
        return "sql_database_chain"

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, db: SQLDatabase, prompt: Optional[BasePromptTemplate] = None,
            **kwargs: Any) -> SQLDatabaseChain:
        """Create a SQLDatabaseChain from an LLM and a database connection.

        *Security note*: Make sure that the database connection uses credentials
            that are narrowly-scoped to only include the permissions this chain needs.
            Failure to do so may result in data corruption or loss, since this chain may
            attempt commands like `DROP TABLE` or `INSERT` if appropriately prompted.
            The best way to guard against such negative outcomes is to (as appropriate)
            limit the permissions granted to the credentials used with this chain.
            This issue shows an example negative outcome if these steps are not taken:
            https://github.com/langchain-ai/langchain/issues/5923
        """
        prompt = prompt or SQL_PROMPTS.get(db.dialect, PROMPT)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, database=db, prompt=prompt, **kwargs)

