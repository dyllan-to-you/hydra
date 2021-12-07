from datetime import datetime
from functools import wraps
from timeit import default_timer as timer
from typing import List, Union
import psutil
import os
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


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    txt = f"[{now()}] ({process.memory_info().rss / 1024 / 1024:.2f}) {' '.join([str(a) for a in arg])}"
    if pbar:
        pbar.write(txt)
    else:
        print(txt)


def write(message, filename="log.txt"):
    f = open(filename, "a")
    f.write(f"[{now()}] {message}\n")
    f.close()


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