import numpy as np
from pobm.obm.general import OverallGeneralMeasures
from pobm.obm.burden import HypoxicBurdenMeasures
from pobm.obm.desat import DesaturationsMeasures

from brainteaser_preprocessing.utils import fillNanIslands, findNanIsland, moving_average, retime, graph, showpreprocess


class SPO2:
    """
    class to compute spo2 related features
    """
    spo2_clean: dict
    spo2_1m: np.array
    sleep_duration: int
    showPlot: bool
    start_offset: int
    save: bool

    def __init__(self, spo2_1m, sleep_duration: int, showPlot=False, start_offset: int = 0, save=False):
        self.spo2_1m = spo2_1m
        self.sleep_duration = sleep_duration
        self.showPlot = showPlot
        self.start_offset = start_offset
        self.save = save

    def run(self):
        self.preprocess()
        if(self.showPlot):
            graph(self.spo2_clean, "SpO2", save=self.save)
        return self.compute()

    def compute(self):
        # # print('----SPO2----')

        overall = OverallGeneralMeasures(90).compute(
            np.fromiter(self.spo2_clean.values(), dtype=float))
        # # print('overall ', overall)
        desat = DesaturationsMeasures().compute(
            np.fromiter(self.spo2_1m.values(), dtype=float))
        # # print('desat ', desat)
        hypo = HypoxicBurdenMeasures(desat.begin, desat.end).compute(
            np.fromiter(self.spo2_1m.values(), dtype=float))
        # # print('hypoxic ', hypo)
        return {
            "AV": overall.AV,
            "MED": overall.MED,
            "Min": overall.Min,
            "SD": overall.SD,
            "RG": overall.RG,
            "P": overall.P,
            "M": overall.M,
            "ZC": overall.ZC,
            "DI": overall.DI,
            "CA": hypo.CA,
            "CT": hypo.CT,
            "POD": hypo.POD,
            "AODmax": hypo.AODmax,
            "AOD100": hypo.AOD100,
            "ODI": desat.ODI
        }

    def preprocess(self) -> np.array:
        """
            Preprocessing of spo2 data
            input: dict{offset, value}
            output: dict{offset, value} - 60s grid with nan values
        """
        grid_spo2 = retime(self.spo2_1m, 60, self.sleep_duration, False)
        filled_spo2 = self.__inputation(grid_spo2)
        avg_spo2 = moving_average(filled_spo2, 3)
        self.spo2_clean = avg_spo2
        if(self.showPlot):
            showpreprocess(grid=grid_spo2, filled=filled_spo2,
                           avged=avg_spo2, title="SpO2 preprocess", start=self.start_offset, save=self.save)
        return avg_spo2

    def __inputation(self, grid_spo2: dict) -> dict:
        [start, end, isle] = findNanIsland(grid_spo2, 60, 60*5)

        filled = fillNanIslands(grid_spo2, start, end)
        return filled
