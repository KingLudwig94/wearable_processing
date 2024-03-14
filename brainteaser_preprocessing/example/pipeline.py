import json
import os
import sys

from brainteaser_preprocessing.pipeline import run_pipeline

"""
Method to run the pipeline on a file passed as first argument in a console call and save the result in a file passed in the second argument
"""
def run_pipeline_with_files():
    try:
        infilename = sys.argv[1]
    except IndexError:
        print("you need to pass in a file name to process")
        print(help)
        sys.exit()
    try:
        outfilename = sys.argv[2]
    except IndexError:
        root, ext = os.path.splitext(infilename)
        outfilename = root + "-features" + ".json"

    print("Running pipeline on file: %s and saving features in %s" % (infilename, outfilename))
    with open(infilename, 'r') as f:
        data = json.load(f)
        out = run_pipeline(data)
    with open(outfilename, 'w') as o:
        o.write(out)
    print("Done")


if __name__ == "__main__":
    run_pipeline_with_files()
