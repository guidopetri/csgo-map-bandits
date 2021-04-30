import pandas as pd
import context_engineering_functions as cef
import sys


def main(datapath):
    context = cef.create_basic_pick_veto_triples(datapath, concat = False, save = False)
    print(context[0].head(20))
    print(context[1].head(20))


if __name__ == "__main__":
    filepath = sys.argv[1]
    main(filepath)

