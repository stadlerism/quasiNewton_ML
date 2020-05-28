import traceback

class IterationCompleteException(Exception):
    def __init__(self):
        traceback.print_stack()
    pass
