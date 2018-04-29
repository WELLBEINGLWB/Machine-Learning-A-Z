import logging
"""
logging levels

CRITICAL 50
ERROR    40
WARNING  30
INFO     20
DEBUG    10
NOTSET   0

default level WARNING sets lower level of logging- means it will log any warnings, errors, criticals
"""
# set config to debug level
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
#formatter
formatter = logging.Formatter('%(levelname)s%(name)s:%(message)s')
# log handler
log_hdlr = logging.FileHandler("calculations.log")
log_hdlr.setFormatter(formatter)
# stream handler
#stream_hdlr = logging.StreamHandler()
#stream_hdlr.setFormatter(formatter)
# add handlers
if not logger.handlers:
   logger.addHandler(log_hdlr)
#logger.addHandler(stream_hdlr)

def add(x, y):
    """Add Function"""
    return x + y

def subtract(x, y):
    """Subtract Function"""
    return x - y

def multiply(x, y):
    """Multiply Function"""
    return x * y

def divide(x, y):
    """Divide Function"""
    try:
       result = x/y
    except ZeroDivisionError:
       logger.exception("Zero division")
    else:
       return result

num_1 = 20
num_2 = 1
add_result = add(num_1, num_2)
logger.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))

sub_result = subtract(num_1, num_2)
logger.debug('Sub: {} - {} = {}'.format(num_1, num_2, sub_result))

mul_result = multiply(num_1, num_2)
logger.warning('Mul: {} * {} = {}'.format(num_1, num_2, mul_result))

div_result = divide(num_1, num_2)
logger.debug('Div: {} / {} = {}'.format(num_1, num_2, div_result))