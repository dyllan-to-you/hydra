from datetime import date


flatten = lambda t: [item for sublist in t for item in sublist]

def now():
    return date.today().strftime("%Y-%m-%d %H:%M:%S")

def printd(*arg):
    print(f"[{now()}]", *arg)

def write(message, filename = 'log.txt'):
    f = open(filename, "a")
    f.write(f"[{now()}] {message}")
    f.close()

def get_closest(num, list):
    return min(list, key=lambda x:abs(x-num))

def get_methods(object, spacing=20):
  methodList = []
  for method_name in dir(object):
    try:
        if callable(getattr(object, method_name)):
            methodList.append(str(method_name))
    except:
        methodList.append(str(method_name))
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
  for method in methodList:
    try:
        print(str(method.ljust(spacing)) + ' ' +
              processFunc(str(getattr(object, method).__doc__)[0:90]))
    except:
        print(method.ljust(spacing) + ' ' + ' getattr() failed')