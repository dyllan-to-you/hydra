from datetime import datetime
from functools import wraps
from time import time
import psutil
import os

process = psutil.Process(os.getpid())


def sanitize_filename(filename):
    keepcharacters = (" ", ".", "_")
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()


def flatten(t):
    return [item for sublist in t for item in sublist]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def printd(*arg):
    print(f"[{now()}] ({process.memory_info().rss / 1024 / 1024:.2f})", *arg)


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


def timeme(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap
