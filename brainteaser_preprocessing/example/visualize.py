import pipeline
import json

""" 
Simple script to visualize the result of the pipeline on a specific input file
"""
with open('data/2022-11-29.json', 'r') as f:
    data = json.load(f)

print(pipeline.run_pipeline(data, False, False))
