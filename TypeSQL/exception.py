import sys
import traceback
from TypeSQL.logger import logging


def error_message_details(error, error_detail: sys):
    _, _, tb = error_detail.exc_info()
    filename = tb.tb_frame.f_code.co_filename
    lineno = tb.tb_lineno
    error = str(error)
    error_message = f"""
                    Error occurred in: {filename} in Line No {lineno} complete Error message: {str(error)}
                    """

    return error_message


class TypeSQLException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)
        self.traceback = traceback.format_exc()

    def __str__(self):
        return str(self.error_message)
