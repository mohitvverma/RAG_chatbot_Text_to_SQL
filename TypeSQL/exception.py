import os
import sys
import traceback
from TypeSQL.logger import logging

def error_message_details(error, error_message: sys):
    _, _, tb = error_message.exc_info()
    filename = tb.tb_frame.f_code.co_filename
    error_message = f"""
                    Error Occured in: {filename} in Line No {tb.tb_lineno} complete Error message: {str(error)}
                    """

    return error_message


class TypeSQLException(Exception):
    def __init__(self, error, error_message: sys):
        super().__init__(error_message)
        self.error = error
        self.error_message = error_message_details(error,error_message = error_message)
        self.traceback = traceback.format_exc()

    def __str__(self):
        return str(self.error)
