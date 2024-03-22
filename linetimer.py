"""
Code cridit: https://github.com/JustasB/linetimer
"""


import time
import timeit
from typing import Union, Optional

UNIT_NANOSECONDS = 'ns'
UNIT_MICROSECONDS = 'us'
UNIT_MILLISECONDS = 'ms'
UNIT_SECONDS = 's'
UNIT_MINUTES = 'm'
UNIT_HOURS = 'h'

time_units = {
    UNIT_NANOSECONDS: 1/1000000000,
    UNIT_MICROSECONDS: 1/1000000,
    UNIT_MILLISECONDS: 1/1000,
    UNIT_SECONDS: 1,
    UNIT_MINUTES: 60,
    UNIT_HOURS: 3600
}


class CodeTimer:

    def __init__(
            self,
            name: str = None,
            silent: bool = False,
            unit: str = UNIT_SECONDS,
            logger_func=None,
            dict_collect=None,
            threshold: Optional[Union[int, float]] = None
    ):
        """
        :param name: A custom name given to a code block
        :param silent: When True, does not print or log any messages
        :param unit: Units to measure time.
                One of ['ns', 'us', 'ms', 's', 'm', 'h']
        :param logger_func: A function that takes a string parameter
                that is called at the end of the indented block.
                If specified, messages will not be printed to console.
        :param dict_collect: Return a dict with key=name, and val=time in `unit`
        :param threshold: A integer or float value. If time taken by code block
                took greater than or equal value, only then log.
                If None, will bypass this parameter.
        """

        self.name = name
        self.silent = silent
        self.unit = unit
        self.logger_func = logger_func
        self.dict_collect = dict_collect
        self.log_str = None
        self.threshold = threshold

    def __enter__(self):
        """
        Start measuring at the start of indent

        :return: CodeTimer object
        """

        self.start = timeit.default_timer()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stop measuring at the end of indent.
        This will run even if the indented lines raise an exception.
        """

        # Record elapsed time in seconds
        self.took = timeit.default_timer() - self.start

        # Convert time into given units
        self.took = (
                self.took
                / time_units.get(self.unit, time_units[UNIT_SECONDS])
        )

        if not self.silent and (
                not self.threshold or self.threshold <= self.took
        ):

            if self.dict_collect is not None:
                self.dict_collect[self.name] = self.took

            else:
                # Craft a log message
                log_message = 'Code block{}took: {:.1f} {}'.format(
                    str(" '" + self.name + "' ") if self.name else ' ',
                    float(self.took),
                    str(self.unit))

                if self.logger_func:
                    self.logger_func(log_message)
                else:
                    print(log_message)


# function decorator style
def linetimer(
        show_args=False,
        name: str = None,
        silent: bool = False,
        unit: str = UNIT_SECONDS,
        logger_func=None,
        threshold: Optional[Union[int, float]] = None
):
    """
    Decorating a function will log how long it took to execute each function call

    Usage:

    @linetimer()
    def foo():
        pass

    "Code block 'foo()' took xxx s"

    @linetimer(show_args=True)
    def foo():
        pass

    :param show_args: When True, will print the parameters passed
            into the decorated function
    :param name: If None, uses the name of the function and show_args value.
            Otherwise, same as CodeTimer.
    :param silent: When True, does not print or log any messages
    :param unit: Units to measure time. One of ['ns', 'us', 'ms', 's', 'm', 'h']
    :param logger_func: A function that takes a string parameter
                that is called at the end of the indented block.
                If specified, messages will not be printed to console.
    :param threshold: A integer or float value. If time taken by code block
                took greater than or equal value, only then log.
                If None, will bypass this parameter.
    :return: CodeTimer decorated function
    """

    def decorator(func):

        def wrapper(*args, **kwargs):

            if name is None:
                if show_args:
                    block_name = func.__name__ + '('

                    def to_str(val):
                        return [val].__str__()[1:-1]

                    # append args
                    if len(args) > 0:
                        block_name += ', '.join([to_str(arg) for arg in args])

                        if len(kwargs.keys()) > 0:
                            block_name += ', '

                    # append kwargs
                    if len(kwargs.keys()) > 0:
                        block_name += ', '.join([
                            k + '=' + to_str(v) for k, v in kwargs.items()
                        ])

                    block_name += ')'

                else:
                    block_name = func.__name__

            else:
                block_name = name

            with CodeTimer(
                name=block_name,
                silent=silent,
                unit=unit,
                logger_func=logger_func,
                threshold=threshold
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator



if __name__ == '__main__':
    collect = {}
    with CodeTimer('test', dict_collect=collect):
        time.sleep(1)
    with CodeTimer('test1', dict_collect=collect):
        time.sleep(1)

    print(collect)
