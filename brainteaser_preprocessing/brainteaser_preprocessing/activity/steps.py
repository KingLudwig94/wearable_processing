import numpy as np
import math

from brainteaser_preprocessing.utils import graph, retime


class Steps:
    """
        Class that calculates general features from step counter.
    """
    steps_per_15m: dict

    steps_clean: dict

    showPlots: bool

    def __init__(self, steps_per_15m: dict, showPlots: bool = False):
        self.steps_per_15m = steps_per_15m
        self.showPlots = showPlots

    def run(self):
        self.preprocess()
        return self.compute()

    def compute(self) -> dict:
        """
            Computation of features of steps
            input: dict{offset, value}
            output: dict{name, value} features
        """
        # print('----STEPS----')
        return self.__step_counter()

    def preprocess(self) -> np.array:
        """
            Preprocessing of steps
            input: dict{offset, value}
            output: dict{offset, value} - 900s grid with nan values
        """
        # retiming on uniform grid

        self.steps_clean = retime(self.steps_per_15m, 900, 86400, True)
        if(self.showPlots):
             graph(self.steps_clean, "Steps")
        return self.steps_clean

    def __step_counter(self):
        """
            step_counter function.
            Remove values under the minimum range,
            considered as non-physiological and remove nan from the array.

            Parameters

            steps: dict (offset, value)

            :return: mean, total number of steps, steps made between 12 Am and 6 Pm,
            steps made between 6 Am and 12 Pm, steps made between 12 Pm and 6 Pm,
            steps made between 6 Pm and 12 Am, daily goal if set.

                * mean_daily_steps: mean of steps made in 24 hours
                * steps: total number of steps made in 24 hours
                * morning_steps: steps made within 0.00 Am and 12 Am
                * midday_steps: steps made within 12 Am and 6 Pm
                * evening_steps: steps made within 6 Pm and 12 Pm

        """

        lowerBound = 0
        values = np.fromiter(self.steps_clean.values(), dtype=float)
        values[values < lowerBound] = lowerBound
        steps_per_15m = [x if (np.isnan(x) == False) else 0 for x in values]

        mean_daily_steps = np.mean(steps_per_15m)

        # print (mean_daily_steps, 'average steps per hour')

        steps = np.sum(steps_per_15m)
        # print (steps, 'steps total')

        split = np.array_split(steps_per_15m, 4)

        morning_steps = np.sum(split[0])
        # print ('Steps from 12 Am to 6 Am: ', morning_steps)

        midday_steps = np.sum(split[1])
        # print ('Steps from 6 Am to 12 Pm: ', midday_steps)

        afternoon_steps = np.sum(split[2])
        # print ('steps from 12 Pm to 6 Pm: ', afternoon_steps)

        evening_steps = np.sum(split[3])
        # print ('Steps from 6 Pm to 12 Am: ', evening_steps)

        # da mettere in forma json corretta
        return {
            "Mean_per_15min": mean_daily_steps,
            "Total": steps,
            "12AM-6AM": morning_steps,
            "6AM-12PM": midday_steps,
            "12PM-6PM": afternoon_steps,
            "6PM-12AM": evening_steps
        }
