import decorators
from time import sleep
import pint
from math import pi

ureg = pint.UnitRegistry()


@decorators.timer
def loop(number: int = 10000) -> None:
    for _ in range(number):
        pass


@decorators.logger
def say(word: str = 'foo') -> None:
    print(word)


@decorators.retry(3)
def divide(n: int = 1, m: int = 2) -> float:
    print(n / m)


@decorators.timeout(3)
def wait(seconds: int = 5) -> None:
    sleep(seconds)


@decorators.set_unit("cm^3")
def volume(radius: float = 1.0, height: float = 1.0) -> float:
    return pi * height * (radius ** 2)


@decorators.repeat(3)
def greet(person: str = 'John') -> None:
    print(f'Greetings, {person}.')


@decorators.val_args(str)
def state(text: str) -> None:
    print(text)


if __name__ == '__main__':
    # loop(50000)
    # say('test')
    '''
    try:
        result = divide(5, 0)
    except RuntimeError as e:
        print(e)
    '''
    # wait(10)
    '''
    result = volume(2.0, 6.0) * ureg(volume.unit)
    print(result)
    print(result.to("cubic meters"))
    print(result.to("cubic decimeters").m)
    '''
    # greet('Peter')
    # state(404)
