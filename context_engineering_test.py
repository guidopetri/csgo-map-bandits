import pandas as pd
import context_engineering_functions as cef
import sys


def main(datapath):
 	cef.create_basic_triples(datapath, save = True)


if __name__ == "__main__":
    filepath = sys.argv[1]
    main(filepath)

