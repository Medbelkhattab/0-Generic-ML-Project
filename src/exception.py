import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
     
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self):
        return self.error_message
    
# HOW THIS EXCEPTION HANDLER WORKS (THE FLOW):
#
# 1. An error occurs in a `try` block somewhere in the project.
#
# 2. The `except` block catches the error `e` and calls `raise CustomException(e, sys)`.
#
# 3. This creates an object of the `CustomException` class. Its `__init__` method is triggered.
#
# 4. Inside `__init__`, the `error_message_detail` function is called.
#
# 5. `error_message_detail` uses the `sys` module to find the exact file name and line number of the error.
#
# 6. It then formats a detailed error string and returns it.
#
# 7. This detailed string is stored in the `self.error_message` attribute of the `CustomException` object.
#
# 8. When the exception is printed or logged, the `__str__` method is called, which returns our custom, detailed error message.

#import sys
# ... rest of your code