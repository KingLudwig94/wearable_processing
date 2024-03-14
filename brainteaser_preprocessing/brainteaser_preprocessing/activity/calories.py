import numpy as np

from brainteaser_preprocessing.utils import retime, graph


class Calories:
    """
        Class that calculates general features from calories counter.
    """
    active_calories: dict

    cals_clean: dict

    age: int
    male: bool
    weight: int
    height: int
    showPlot: bool

    def __init__(self, active_calories: dict, age: int, male: bool, weight: int, height: int, showPlot: bool = False):
        self.active_calories = active_calories
        self.male = male
        self.weight = weight
        self.height = height
        self.age = age
        self.showPlot = showPlot


    def run(self):
        self.preprocess()
        return self.compute()

    def compute(self) -> dict():
        """
            Computation of features of calories data
            input: dict{offset, value}
            output: dict{name, value} features
        """

        # print('----CALORIES----')
        return self.__total_calories()

    def preprocess(self) -> np.array:
        """
            Preprocessing of calories data
            input: dict{offset, value}
            output: dict{offset, value} - 900s grid with nan values
        """
        # retime on a uniform grid
        self.cals_clean = retime(self.active_calories, 900, 86400, True)
        if(self.showPlot):
            graph(self.cals_clean, "Calories")
        return self.cals_clean

    def __bmr_evaluation(self):
        """
            bmr_evaluation function.
            Calculate the BMR.

            parameter : age, gender, weight and height.

            return: bmr evaluation result

            * bmr_result: basal metabolic rate of a subject
        """

        # calculate the BMR

        if self.male:
            c1 = 5
        else:
            c1 = -161

        hm = 6.25 * self.height
        wm = 10 * self.weight
        am = 5 * self.age
        # Mifflin-St Jeor Equation
        bmr_result = c1 + hm + wm - am
        # print(int(bmr_result), ' basal calories')
        return bmr_result

    def __active_calories_counter(self, calories: np.array):
        """
            Calories_counter function.
            Clean the signals removing values under a minimum range,
            considered as non-physiological and removing nan from the array.

            :param: 1-d array, of shape (N,) where n is the length of the signal

            :return: processed signal, 1-d numpy array. Active calories mean.

            * tot_active_cal: sum of active calories burned in 24 hours
            * mean_actie_calories: mean of active calories burned in 24 hours
         """

        lowerBound = 0
        calories[calories < lowerBound] = lowerBound
        active_calories = [x for x in calories if np.isnan(x) == False]

        tot_active_cal = np.sum(active_calories, dtype=int)
        mean_active_calories = np.mean(active_calories)
        # print(tot_active_cal, ' active calories')
        return [tot_active_cal, mean_active_calories]

    # def basal_calories(self):
    #     """"
    #     Basal resting evaluation function.
    #     Clean the signals removing values under a minimum range,
    #     considered as non-physiological and removing nan from the array.

    #     :param self: 1-d array, of shape (N,) where n is the length of the signal

    #     :return: processed signal, 1-d numpy array. Basal resting mean.

    #         * tot_basal resting: sum of the calories burned in 24 hours due to basal resting
    #         * mean_basal_resting: mean of calories burned in 24 hours due to basal resting

    #     """

    #     self.basal_resting[self.basal_resting == 0] = self.__bmr_evaluation()

    #     basal_resting_bound = 0
    #     self.basal_resting[self.basal_resting < basal_resting_bound] = self.__bmr_evaluation()
    #     basal_resting = [x for x in self.basal_resting if np.isnan (x) == False]

    #     tot_basal_resting = np.sum(basal_resting)
    #     mean_basal_resting = np.mean (basal_resting)

    #     # print ('Basal resting mean is: ', mean_basal_resting)

    #     return (tot_basal_resting, mean_basal_resting)

    def __total_calories(self) -> dict():

        # TOTAL CALORIES: active calories + basal reasting
        """
            Total calories counter.
            Sum active calories and total calories burnt in 24 hours.

            :param: signal, two 1-d array, of shape (N,) where n is the length of the signal

            :return: processed signal, 1-d numpy array. Total calories sum. Total calories mean.

                * total_calories: sum of calories burned due to both physical activity and basal resting
                * mean_total calories: mean of calories burned due to both physical activity and basal resting

        """
        active_calories = int(self.__active_calories_counter(
            np.fromiter(self.cals_clean.values(), dtype=float))[0])
        basal_cals = self.__bmr_evaluation()
        total_calories = active_calories + basal_cals
        # print (total_calories, 'total calories')

        return {
            "total": total_calories,
            "active": int(active_calories),
            "basal": int(basal_cals)
        }
