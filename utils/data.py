import types
import datetime
from typing import Any, Union, get_origin, get_args
import inspect


def to_db_types(tp: Any) -> str:
    """
    Converts Python types to ClickHouse data type strings, handling various type annotations
    and checking type inheritance for third-party library types.
    """
    # Handle GenericAlias types like list[str], list[int], etc.
    if isinstance(tp, types.GenericAlias) or (hasattr(tp, '__origin__') and tp.__origin__ is not Union):
        origin = get_origin(tp)
        args = get_args(tp)

        if origin is list:
            if len(args) != 1:
                raise ValueError(f"List type must have exactly one argument, got {args}")
            inner_type = to_db_types(args[0])
            return f"Array({inner_type})"

        raise ValueError(f"Unsupported generic container: {origin}")

    # Handle optional/nullable types (Optional[T] is Union[T, None])
    if get_origin(tp) is Union and type(None) in get_args(tp):
        actual_type = next(arg for arg in get_args(tp) if arg is not type(None))
        return f"Nullable({to_db_types(actual_type)})"

    if inspect.isclass(tp):
        # Numeric types
        if issubclass(tp, int):
            return 'Int64'
        if issubclass(tp, float):
            return 'Float64'

        # String and binary types
        if issubclass(tp, (str, bytes)):
            return 'String'

        # Date/time types
        if issubclass(tp, datetime.date):
            return 'Date' if tp is datetime.date else 'DateTime'

        # Handle numpy types if available
        try:
            import numpy as np
            if issubclass(tp, np.integer):
                return 'Int64'
            if issubclass(tp, np.floating):
                return 'Float64'
            if issubclass(tp, np.bool_):
                return 'Bool'
            if issubclass(tp, np.datetime64):
                return 'DateTime'
        except ImportError:
            pass

    raise ValueError(f"Unsupported type: {tp}")


if __name__ == "__main__":
    # Basic types
    print("Basic types:")
    print("int:", to_db_types(int))  # Int64
    print("float:", to_db_types(float))  # Float64
    print("str:", to_db_types(str))  # String
    print("bool:", to_db_types(bool))  # UInt8
    print("date:", to_db_types(datetime.date))  # Date
    print("datetime:", to_db_types(datetime.datetime))  # DateTime
    print("bytes:", to_db_types(bytes))  # String
    print()

    # Container types
    print("Container types:")
    print("list[str]:", to_db_types(list[str]))  # Array(String)
    print("list[list[float]]:", to_db_types(list[list[float]]))  # Array(Array(Float64))
    print("Optional[int]:", to_db_types(Union[int, type(None)]))  # Nullable(Int64)
    print()

    # Nested generic types
    print("Nested generics:")
    print("list[Optional[float]]:", to_db_types(list[Union[float, None]]))  # Array(Nullable(Float64))
    print("Optional[list[str]]:", to_db_types(Union[list[str], None]))  # Nullable(Array(String))
    print()

    # Third-party types (with safety checks)
    print("Third-party types:")
    try:
        import numpy as np

        print("np.int32:", to_db_types(np.int32))  # Int64
        print("np.float64:", to_db_types(np.float64))  # Float64
        print("np.bool_:", to_db_types(np.bool_))  # UInt8
        print("np.datetime64:", to_db_types(np.datetime64))  # DateTime
    except ImportError:
        print("Numpy tests skipped")

    try:
        import pandas as pd

        print("pd.Timestamp:", to_db_types(pd.Timestamp))  # DateTime
    except ImportError:
        print("Pandas tests skipped")
    print()

    # Type inheritance tests
    print("Type inheritance:")


    class MyInt(int):
        pass


    class MyFloat(float):
        pass


    class MyStr(str):
        pass


    print("MyInt:", to_db_types(MyInt))  # Int64
    print("MyFloat:", to_db_types(MyFloat))  # Float64
    print("MyStr:", to_db_types(MyStr))  # String
    print()

    # Edge cases and error handling
    print("Error handling:")
    try:
        to_db_types(list[int, str])
    except ValueError as e:
        print("Multi-arg list error:", e)

    try:
        to_db_types(dict)
    except ValueError as e:
        print("Unsupported type (dict):", e)

    try:
        to_db_types(set)
    except ValueError as e:
        print("Unsupported type (set):", e)

    try:
        to_db_types(Union[int, str])
    except ValueError as e:
        print("Non-optional union error:", e)
    print()

    # Test invalid Optional syntax (should still work)
    print("Union[float, None]:", to_db_types(Union[float, None]))  # Nullable(Float64)