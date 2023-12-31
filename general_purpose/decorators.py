import time
import functools
import datetime
import threading
import tracemalloc
from typing import Callable
# from functools import lru_cache


def timer(function: Callable):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        t1 = time.perf_counter()
        result = function(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        t2 = time.perf_counter()
        print(f'{"-" * 40}')
        print(f'args: \t\t\t\t {args}')
        print(f'kwargs: \t\t\t {kwargs}')
        print(f'memory usage: \t\t {current / 10 ** 6:.6f} MB')
        print(f'peak memory usage: \t {peak / 10 ** 6:.6f} MB')
        print(f'"{function.__name__}" ran in \t\t {round(t2 - t1, 6)} seconds')
        print(f'result: \t\t\t {result}')
        print(f'{"-" * 40}')
        tracemalloc.stop()
        return result
    return wrapper


def logger(function: Callable):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with open('./logger.txt', 'a') as file:
            file.write(
                f'Called "{function.__name__}" with {args} at {datetime.datetime.now()}.\n')
        file.close()
        result = function(*args, **kwargs)
        return result
    return wrapper


def retry(max_retries: int = 3, exception_to_catch: Exception = ZeroDivisionError, delay_between_retries: int = 1):
    def decorator(function: Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    result = function(*args, **kwargs)
                    return result
                except exception_to_catch as e:
                    attempts += 1
                    print(
                        f'Retry {attempts}/{max_retries} after {delay_between_retries} {"seconds" if delay_between_retries > 1 else "second"} due to {e}.')
                    time.sleep(delay_between_retries)
            raise RuntimeError(
                f'Function "{function.__name__}" failed after {max_retries} retries.')
        return wrapper
    return decorator


def timeout(seconds: int = 3):
    def decorator(function: Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            result = None
            exception = None

            def target():
                nonlocal result, exception
                try:
                    result = function(*args, **kwargs)
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                thread.join()
                raise TimeoutError(
                    f'{function.__name__} took longer than {seconds} {"seconds" if seconds > 1 else "second"} to execute.')

            if exception:
                raise exception

            return result
        return wrapper
    return decorator


def set_unit(unit: str = 'm/s', type: str = '2d-vector'):
    def decorator(function: Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            wrapper.unit = unit
            wrapper.type = type
            return result
        return wrapper
    return decorator


def repeat(times: int = 3):
    def decorator(function: Callable):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                function(*args, **kwargs)
        return wrapper
    return decorator
