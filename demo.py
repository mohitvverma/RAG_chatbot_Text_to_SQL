import os
import hashlib
import logging
from typing import Dict

from TypeSQL.logger import logging
from TypeSQL.pipeline.generative_pipeline import SQL2TextPipeline
from TypeSQL.settings import obtain_configs
from TypeSQL.prompt import SQL_MSG, AI_MSG


def initialize_session_state() -> Dict:
    """Initialize the session state for local application"""
    logging.info("Initializing session state")
    model_configs = obtain_configs()
    prompt_configs = {
        "sql_msg": SQL_MSG,
        "ai_msg": AI_MSG,
        # Add other prompt configurations as needed
    }
    return model_configs, prompt_configs


def main() -> None:
    model_configs, prompt_configs = initialize_session_state()
    pipeline = SQL2TextPipeline()
    pipeline.setup(model_configs=model_configs, prompt_configs=prompt_configs)
    # Add your local MySQL database connection and testing code here
    # For example:
    # db = MySQLDatabaseConnection("localhost", "username", "password", "database_name")
    # pipeline.test_with_db(db)


if __name__ == "__main__":
    main()
