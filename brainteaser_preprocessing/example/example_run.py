from brainteaser_preprocessing.pipeline import *
import json

# import data from json
file = 'data/20-03-22.json'
with open(file) as f:
  data = json.load(f)

# get json string as output
features = run_pipeline(data)
print(features)
