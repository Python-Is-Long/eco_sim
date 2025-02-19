import traceback

def log_traceback():
    a = traceback.extract_stack()
    print(''.join(traceback.format_list(a)))
    

def some_function():
    log_traceback()
    print("Function executed.")

some_function()

try:
    x = 1 / 0  # This raises ZeroDivisionError
except ZeroDivisionError as e:
    raise ValueError("Invalid operation") from e
