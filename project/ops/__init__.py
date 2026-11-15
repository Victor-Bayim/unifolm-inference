# ops/__init__.py
from . import my_gelu_op

def gelu_and_mul(out, input):
    return my_gelu_op.gelu_and_mul(out, input)