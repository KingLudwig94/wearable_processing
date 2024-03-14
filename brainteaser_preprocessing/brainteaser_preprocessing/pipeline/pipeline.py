"""
Method to run the pipeline on a dictionary passed as first argument
Input: a json list with one element in the format defined in the deliverable 5.1
Output: a json formatted string
"""
import brainteaser_preprocessing
import json
from matplotlib.pyplot import show
import warnings

def run_pipeline(data, showPlots: bool = False, savePlots: bool = False):
    warnings.filterwarnings('ignore')
    data = data[0]
    # create output dictionary
    out = dict()
    out['baseDate'] = data['base_date']
    out['baseDateOffset'] = data['base_date_offset']

    # STEPS
    st = data["steps"]["data"]
    if (st != None):
        st2 = {}
        for row in st:
            st2[row['offset']] = row['value']
        st = st2
        out['steps'] = brainteaser_preprocessing.activity.Steps(st, showPlots).run()

    if 'activity' in data:
        # CALORIES
        cal = data["activity"]["data"]
        if (cal != None):
            cal2 = {}
            for row in cal:
                cal2[row['offset']] = row['active_kilocalories']

            # TODO: add variables of patient for height, weight, age, gender
            out['calories'] = brainteaser_preprocessing.activity.Calories(
                cal2, data["age"], data["gender"] == "MALE", 80, 185, showPlots).run()

    # HR
    hr = data['heart_rate']['data']
    if (hr != None):
        hr2 = {}
        for row in hr:
            hr2[row['offset']] = row['value']
        hr = hr2
        if (data['heart_rate']['min_heart_rate'] != data['heart_rate']['max_heart_rate']):
            out['heartRate'] = brainteaser_preprocessing.hrv.HRV(hr, showPlots, save=savePlots).run()
        out['heartRate']['baseline'] = data['heart_rate']['resting_heart_rate']
        out['heartRate']['minimum'] = data['heart_rate']['min_heart_rate']
        out['heartRate']['maximum'] = data['heart_rate']['max_heart_rate']
        out['heartRate']['mean'] = data['heart_rate']['mean_heart_rate']

    if 'sleep' in data:
        if 'spo2' in data['sleep']:
            # SPO2
            spo2 = data['sleep']['spo2']
            if (spo2 != None):
                spo22 = {}
                for row in spo2:
                    spo22[row['offset']] = row['value']
                spo2 = spo22
                out["spo2"] = (brainteaser_preprocessing.spo2.SPO2(spo2, data['sleep']['duration'],
                                       showPlots, data['sleep']['start_offset'], save=savePlots).run())

        if 'sleep_level_map' in data['sleep']:
            # SLEEP
            sleep = data['sleep']['sleep_level_map']
            if (sleep != None):
                out['sleep'] = {
                    "duration": data['sleep']['duration'],
                    "startOffset": data['sleep']['start_offset'],
                    "unmeasurableSleepDurationInSeconds": data['sleep']['unmeasurable_sleep_duration_in_seconds'],
                    "deepSleepDurationInSeconds": data['sleep']['deep_sleep_duration_in_seconds'],
                    "lightSleepDurationInSeconds": data['sleep']['light_sleep_duration_in_seconds'],
                    "remSleepInSeconds": data['sleep']['rem_sleep_in_seconds'],
                    "awakeDurationInSeconds": data['sleep']['awake_duration_in_seconds']
                }

        if 'respiration' in data['sleep']:
            # RESPIRATION
            rsp = data['sleep']['respiration']
            if (rsp != None):
                rsp2 = {}
                for row in rsp:
                    rsp2[row['offset']] = row['value']
                rsp = rsp2
                out['respiration'] = brainteaser_preprocessing.respiration.RSP(
                    rsp, data['sleep']['duration'], showPlots, data['sleep']['start_offset'], save=savePlots).run()
    if showPlots:
        show()

    # export features to json string
    return json.dumps(out, allow_nan=True, indent=4)
