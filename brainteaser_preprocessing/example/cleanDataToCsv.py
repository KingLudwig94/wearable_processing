from pandas import DataFrame, Series
import brainteaser_preprocessing.hrv
import brainteaser_preprocessing.activity
from brainteaser_preprocessing.utils import retime

"""
Method to export a csv file with all the preprocessed timeseries
"""
def toCsV(data):
    data = data[0]

    offsets = range(0, 3600*24, 15)

    hr = data['heart_rate']['data']
    if(hr != None):
        hr2 = {}
        for row in hr:
            hr2[row['offset']] = row['value']
        hr = hr2
        hr = brainteaser_preprocessing.hrv.HRV(hr).preprocess()
    hrsseries = Series(hr)

    stress = data['stress']['data']
    if(stress != None):
        stress2 = {}
        for row in stress:
            stress2[row['offset']] = row['value']
        stress = stress2
        stress = retime(stress,15,3600*24)
    stressseries = Series(stress)

    step = data['steps']['data']
    if(step != None):
        step2 = {}
        for row in step:
            step2[row['offset']] = row['value']
        step = step2
        step = brainteaser_preprocessing.activity.Steps(step).preprocess()
    stepseries = Series(step)

    cal = data['activity']['data']
    if(cal != None):
        cal2 = {}
        for row in cal:
            cal2[row['offset']] = row['active_kilocalories']
        cal = cal2
        cal = brainteaser_preprocessing.activity.Calories(cal, 1, True, 1, 1).preprocess()
    calseries = Series(cal)

    met = data['activity']['data']
    if(met != None):
        met2 = {}
        for row in met:
            met2[row['offset']] = row['met']
        met = met2
        met = retime(met,15,3600*24)
    metseries = Series(met)


    df = DataFrame({'hr': hrsseries, 'stress': stressseries, 'step': stepseries, 'activecal': calseries, 'met': metseries})

    return df.to_csv(header=True, index=True)



