import pandas as pd
import context_engineering_functions as cef
import sys


def main(datapath):
    context = cef.create_basic_triples(datapath, save = False)
    print(context.head(20))


if __name__ == "__main__":
    filepath = sys.argv[1]
    main(filepath)

