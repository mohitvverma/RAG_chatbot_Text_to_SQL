import os
import hashlib

import streamlit as st
from TypeSQL.logger import logging
from TypeSQL.exception import TypeSQLException

from TypeSQL.style import markdown_style
from TypeSQL.prompt import SCHEMA_PROMPT, SCHEMA_MODE_MD


def initialize_session_state() -> None:
    """
    Initialize the session state for streamlit applications.
    :return:
    """
    config_keys = ['db_type', 'db_name_mysql', 'db_user', 'db_password', 'db_host', 'db_port']

    for config_key in config_keys:
        st.session_state.setdefault(config_key, "")

    st.session_state.setdefault("db_name_sqlite", 'Chinook.db')
    st.session_state.setdefault("db_schema", SCHEMA_PROMPT)
    st.session_state.setdefault("isdbSchemaMode", False)
    st.session_state.setdefault("isConfigChange", False)


def change_mode() -> dict:
    """
    Choose mode between standard mode and schema mode.
    :return:
    """
    isdbSchemaMode = st.toggle("Database Schema Mode", help="Toggle ON to activate database schema mode")
    configs = {"isdbSchemaMode": isdbSchemaMode}
    if isdbSchemaMode != st.session_state['isdbSchemaMode']:
        st.session_state["isConfigChange"] = True
        logging.info("<blue>Changing mode...</blue>")

    st.session_state["isdbSchemaMode"] = isdbSchemaMode
    return configs


def db_schema_mode(configs: dict) -> dict:
    """
    User can run schema mode with their own database
    :param configs:
    :return:
    """
    db_schema = st.text_area(
        "Database Schema",
        height=600,
        value=st.session_state["db_schema"],
        key="db_schema_input",
        on_change=lambda: (setattr(st.session_state, "db_schema", st.session_state["db_schema_input"]),
                           setattr(st.session_state, "isConfigChange", True))

                 )

    configs.update({"db_schema": db_schema})
    return configs


def standard_mode(configs: dict) -> dict:
    """
    Choose between sqllite and mysql mode.
    :param configs:
    :return:
    """
    select_db = st.sidebar.selectbox(
        "Database Type",
        ("SQLite", "MySQL", 'PostgresSQL')
    )

    if select_db == "SQLite":
        db_type = "SQLite"
        db_name_sqlite = st.text_input(
            "Database FileName",
            value=st.session_state["db_name_sqlite"],
            key="db_name_sqlite_input",
            on_change=lambda: (setattr(st.session_state, "db_name_sqlite",
                                       st.session_state["db_name_sqlite"]),
                               setattr(st.session_state, "isConfigChange", True))
        )
        config_values = [db_type, db_name_sqlite]
        new_keys = ["db_type", 'db_name_sqlite']
        db_config = dict(zip(new_keys, config_values))

        if st.session_state["isConfigChange"]:
            st.toast("Please ensure database is in 'sqlite' directory.", icon="🚨")

    else:
        db_type = "MySQL"
        db_name_mysql = st.text_input(
            "Database FileName",
            value=st.session_state["db_name_mysql"],
            key="db_name_mysql_input",
            on_change=lambda: (setattr(st.session_state, "db_name_mysql",st.session_state["db_name_mysql_input"]),
                               setattr(st.session_state, "isConfigChange", True)))

        username = st.text_input(
            "Database username",
            value=st.session_state["username"],
            key="username_input",
            on_change=lambda: (setattr(st.session_state, "username", st.session_state["username_input"]),
                               setattr(st.session_state, "isConfigChange", True)))

        password = st.text_input(
            "Database password",
            value=st.session_state["password"],
            key="password_input",
            on_change=lambda: (setattr(st.session_state, "password", st.session_state["password_input"]),
                               setattr(st.session_state, "isConfigChange", True))
        )

        hostname = st.text_input(
            "Database hostname",
            value=st.session_state["hostname"],
            key="hostname_input",
            on_change=lambda: (setattr(st.session_state, "hostname", st.session_state["hostname_input"]),
                               setattr(st.session_state, "isConfigChange", True))
        )

        port = st.text_input(
            "Database port",
            value=st.session_state["port"],
            key="port_input",
            on_change=lambda: (setattr(st.session_state, "port", st.session_state["port_input"]),
                               setattr(st.session_state, "isConfigChange", True))
        )

        config_values = [db_type, db_name_mysql, username, password, hostname, port]
        new_keys = ["db_type", 'db_name_mysql', 'username', 'password', 'hostname','port']
        db_config = dict(zip(new_keys, config_values))

    configs.update(db_config)
    return configs


def update_logs(configs) -> None:
    """
    print logs in log file if config was changed
    :param configs:
    :return:
    """
    config_logs = configs.copy()

    if 'password' in config_logs:
        password = config_logs['password']
        config_logs['password'] = hashlib.sha256(password.encode()).hexdigest()

    if st.session_state["isConfigChange"]:
        logging.info("<blue>Updating logs...</blue>")
        logging.info(f"<yellow>New configs: {config_logs}</yellow>")
        st.session_state["isConfigChange"] = False


def setup_ChatUI_sidebar() -> dict:
    """
    Setup for streamlit Chat UI Sidebar
    :return:
    """
    st.title("🤖 Chat With Your :green[Data]", anchor=False)
    markdown_style()
    initialize_session_state()
    with st.sidebar:
        configs = change_mode()
        if configs["isConfigChange"]:
            configs = db_schema_mode(configs)
        else:
            configs = standard_mode(configs)

    if configs["isSchemaMode"]:
        st.markdown(SCHEMA_MODE_MD)
    update_logs(configs)

    return configs
