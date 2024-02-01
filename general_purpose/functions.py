from random import sample
from string import ascii_lowercase, ascii_uppercase, digits, punctuation
from collections import Counter, OrderedDict
from typing import Dict, Any, Iterable, List, Callable
import heapq


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dictionaries(result[key], value)
            else:
                result[key] = value
    return result


def unique_elements(_list: Iterable[Any]) -> list[Any]:
    return list(OrderedDict.fromkeys(_list))


def count_occurrences(_list: Iterable[Any]) -> Dict[Any, int]:
    return dict(Counter(_list))


def chunk_file(input_file: str, chunk_size: int) -> List[str]:
    with open(input_file, 'r') as file:
        content = file.read()
        return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]


def find_common_elements(list1: List[Any], list2: List[Any]) -> List[Any]:
    return list(set(list1) & set(list2))


def reverse_dictionary(dict: Dict[Any, Any]) -> Dict[Any, Any]:
    return {v: k for k, v in dict.items()}


def is_sorted(_list: List[Any]) -> bool:
    return all(_list[i] <= _list[i + 1] for i in range(len(_list) - 1))


def generate_password(length: int = 16) -> str:
    characters = ascii_lowercase + ascii_uppercase + digits + punctuation
    return ''.join(sample(characters, length))


class TaskScheduler:
    def __init__(self) -> None:
        self._task_queue = []

    def schedule(self, task: Callable, priority: int) -> None:
        heapq.heappush(self._task_queue, (priority, task))

    def execute_tasks(self) -> None:
        while self._task_queue:
            priority, task = heapq.heappop(self._task_queue)
            task()
            print(f"Executing task: {task.__name__} with priority {priority}.")
