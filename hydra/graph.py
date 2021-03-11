import argparse
import pickle

parser = argparse.ArgumentParser('Graph Me!')
parser.add_argument('path', type=str)

def load(path):
    return pickle.load(open(path, 'rb'))

def draw():
    args = parser.parse_args()
    portfolio = load(args.path)
    print(portfolio)