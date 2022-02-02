import functools
import threading
from datetime import datetime
from functools import wraps
from timeit import default_timer as timer
from typing import List, Union
import psutil
import os
import numpy as np
import pandas as pd

process = psutil.Process(os.getpid())


def str_to_datetime(str) -> datetime:
    return (
        datetime.strptime(str, "%Y-%m-%d %H:%M:%S")
        if not isinstance(str, datetime)
        else str
    )


def sanitize_filename(filename):
    keepcharacters = (" ", ".", "_")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()


def flatten(t):
    return [item for sublist in t for item in sublist]


def now(str=True):
    dt = datetime.now()
    if not str:
        return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def striped_chunk_list(l: List, n: int):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def chunk_list(l: List, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def get_mem(unit=None):
    mem = process.memory_info().rss
    if unit is None:
        return mem
    if unit == "KB":
        return mem / 1024
    if unit == "MB":
        return mem / 1024 ** 2
    if unit == "GB":
        return mem / 1024 ** 3


def printd(*arg, pbar=None):
    txt = f"[{now()}] ({get_mem('MB'):.2f}MB) {' '.join([str(a) for a in arg])}"
    if pbar:
        pbar.write(txt)
    else:
        print(txt)


filemap = dict()


def write(*message, filename="log.txt"):
    msg = " ".join([str(a) for a in message])
    msg = f"[{now()}] {msg}\n"
    # print(msg)
    # return
    fp = filemap.get(filename, None)
    if fp is None:
        fp = open(filename, "a")
        filemap[filename] = fp
    # fp = open(filename, "a")
    fp.write(msg)
    fp.flush()
    # typically the above line would do. however this is used to ensure that the file is written
    os.fsync(fp.fileno())


def get_closest(num, list):
    return min(list, key=lambda x: abs(x - num))


def get_methods(object, spacing=20):
    methodList = []
    for method_name in dir(object):
        try:
            if callable(getattr(object, method_name)):
                methodList.append(str(method_name))
        except:
            methodList.append(str(method_name))
    processFunc = (lambda s: " ".join(s.split())) or (lambda s: s)
    for method in methodList:
        try:
            print(
                str(method.ljust(spacing))
                + " "
                + processFunc(str(getattr(object, method).__doc__)[0:90])
            )
        except:
            print(method.ljust(spacing) + " " + " getattr() failed")


def timeme(fn):
    @wraps(fn)
    def wrap(*args, **kw):
        ts = timer()
        result = fn(*args, **kw)
        te = timer()

        elapsed = te - ts
        # print(
        #     "func:%r args:[%r, %r] took: %2.4f sec" % (fn.__name__, args, kw, elapsed)
        # )
        printd("func:%r took: %2.4f sec" % (fn.__name__, elapsed))
        return result

    return wrap


def split_dupes(df: Union[pd.DataFrame, pd.Series]):
    duplicates = df
    sets = []
    while len(duplicates):
        uniques = duplicates[~duplicates.index.duplicated(keep="first")]
        duplicates = duplicates[duplicates.index.duplicated(keep="first")]
        sets.append(uniques)
    return sets


def mem_used(total=psutil.virtual_memory().total):
    return total - psutil.virtual_memory().available


def debounce(wait_time):
    """
    Decorator that will debounce a function so that it is called after wait_time seconds
    If it is called multiple times, will wait for the last call to be debounced and run only this one.
    See the test_debounce.py file for examples
    """
    write(f"Initializing debounced function {wait_time}ms")

    def decorator(function):
        write(f"Decorated {function.__name__}")

        def debounced(*args, **kwargs):
            debounced._counter += 1
            write(f"{function.__name__} call #{debounced._counter}")

            def call_function():
                debounced._timer = None
                write(f"!!! Debounced fn {function.__name__} executing {wait_time}")
                debounced._counter = 0
                return function(*args, **kwargs)

            if debounced._timer is not None:
                write(f"Timer is not none. Canceling old timer")
                debounced._timer.cancel()

            write(f"Initializing new timer")
            debounced._timer = threading.Timer(wait_time, call_function)
            write(f"Starting new timer")
            debounced._timer.start()
            write("Timer started!")

        debounced._timer = None
        debounced._counter = 0
        # debounced._totalTime = 0
        return debounced

    return decorator


def rsetattr(obj, attr, val, separator="."):
    pre, _, post = attr.rpartition(separator)
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427


def rgetattr(obj, attr, *args, separator="."):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split(separator))


def getClosestIndex_np(arr, val):
    return np.argmin(np.abs(np.array(arr) - val))
