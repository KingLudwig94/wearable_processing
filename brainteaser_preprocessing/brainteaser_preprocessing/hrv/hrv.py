from curses import window
from locale import windows_locale
import numpy as np
from pandas import DataFrame, concat
import scipy.stats
import hrvanalysis as hv
from brainteaser_preprocessing.utils import fillNanIslands, find_consecutive, findNanIsland, moving_average, polifit_nan, r2, retime, signal_zerocrossings, graph, showpreprocess


class HRV:
    """
    Class for heart rate variability feature extraction
    """

    hr_15s: dict
    showPlots: bool
    save: bool

    def __init__(self, hr_15s, showPlots=False, save=False):
        self.hr_15s = hr_15s
        self.showPlots = showPlots
        self.save = save

    def run(self):
        self.preprocess()
        if(self.showPlots):
            graph(self.hrv_clean, "Heart Rate", save = self.save)
        return self.compute(data = self.hrv_clean, dataval=np.fromiter(self.hrv_clean.values(), dtype=float))

    def computeWindow(self, data: dict, windowLength: int, windowSeparation: int, sr) -> str:
        items = np.fromiter(data.values(), dtype=float)
        offsets = list(data.keys())
        
        n = int(windowLength/sr)
        out = DataFrame()
        windowSeparation = int(windowSeparation/sr)
        i=0
        while(i*windowSeparation+n<len(items)):
            #print(i)
            res = DataFrame(self.compute(dict(zip(offsets[i*windowSeparation:i*windowSeparation+n],items[i*windowSeparation : i*windowSeparation + n])), items[i*windowSeparation : i*windowSeparation + n]), index = [offsets[i]])
            out = concat([out, res])
            i += 1
        return out.to_csv()
            
    def compute(self, data: dict, dataval: np.array ) -> dict:
        # print('----HR----')
        
        hr_mean = np.mean(dataval)
        hr_STD = np.std(dataval[np.isfinite(dataval)])
        out = {}
        maxtime = int(max(data, key=lambda k: data[k] if np.isfinite(
            data[k]) else 0))
        mintime = int(min(data, key=lambda k: data[k] if np.isfinite(
            data[k]) else 1000))
        p = polifit_nan(data)[0]
        p2 = polifit_nan(data, deg=2)
        out.update({'linearTrend': p, 'quadraticTrend': p2[0]})
        if(p2[0] != None):
            yhat = np.polyval(p2, list(data.keys()))
            r = r2(yhat, dataval)
            out.update({'r2': r,})

        # bbi
        bbidict = {k: 60000/v for k, v in data.items()}  # milliseconds
        bbi = np.fromiter(bbidict.values(), dtype=float)
        bbinan = bbi[np.isfinite(bbi)]
        nn = np.diff(bbi)
        nnnan = nn[np.isfinite(nn)]
        absNN = np.abs(nn)
        meanNN = np.mean(bbinan)
        sdnn = np.std(bbinan)
        [sdnna1, sdnni1] = self.segmenting(bbi, 4)
        [sdnna2, sdnni2] = self.segmenting(bbi, 8)
        [sdnna5, sdnni5] = self.segmenting(bbi, 20)

        # # maximum frequency
        # time_step=1/15 #Hz
        # #array of frequencies
        # freqs = np.fft.fftfreq(len(bbinan), time_step)
        # ff = np.fft.fft(bbinan)
        # trange = np.linspace(0, 1/15, len(bbinan))
        # max_freq = freqs[ff==ff.max]
        # #minimum frequency
        # min_freq = min(freqs)
        # plot.plot(trange, np.abs(ff))
        if(len(bbinan)>1):
            time_domain_features = hv.get_time_domain_features(bbinan)
            freq_domain_features = hv.get_frequency_domain_features(
                bbinan, sampling_frequency=1)
            geometrical_features = hv.get_geometrical_features(bbinan)
            poincare_features = hv.get_poincare_plot_features(bbinan)
            csi_cvi = hv.get_csi_cvi_features(bbinan)

        medianBB = np.nanmedian(absNN)
        madBB = scipy.stats.median_abs_deviation(bbinan)
        hcvnn = madBB/medianBB
        iqrange = scipy.stats.iqr(bbinan)
        pnn50 = 100*len(list(x for x in nnnan if abs(x) > 50))/len(nnnan) if len(nnnan)>0 else None
        pnn20 = 100*len(list(x for x in nnnan if abs(x) > 20))/len(nnnan) if len(nnnan)>0 else None
        out.update({'mean': hr_mean, 'std': hr_STD, 'maxTime': maxtime, 'minTime': mintime,
               'meanNN': meanNN, 'sdNN': sdnn, 'sdaNN1': sdnna1, 'sdNNI1': sdnni1, 'sdaNN2': sdnna2, 'sdNNI2': sdnni2, 'sdaNN5': sdnna5, 'sdNNI5': sdnni5,
               'medianNN': medianBB, 'madNN': madBB, 'hcvNN': hcvnn, 'iqrNN': iqrange, 'pNN50': pnn50, 'pNN20': pnn20})
        if(len(bbinan)>1):
            out.update({
                'sdsd': time_domain_features['sdsd'], 'rmssd': time_domain_features['rmssd'], 'cvNN': time_domain_features['cvnni'], 'cvsd': time_domain_features['cvsd'],
                'tiNN': geometrical_features['tinn'], 'HTI': geometrical_features['triangular_index'], 'sd1': poincare_features['sd1'], 'sd2': poincare_features['sd2'], 'sd1sd2': poincare_features['ratio_sd2_sd1'],
                'cvi': csi_cvi['cvi'], 'csi': csi_cvi['csi'], 'csiModified': csi_cvi['Modified_csi']})
            out.update(freq_domain_features)
            self._hrv_nonlinear_poincare_hra(bbinan, out)
            self._hrv_nonlinear_fragmentation(bbinan, out)
        return out

    def preprocess(self) -> np.array:
        """
            Preprocessing of hrv
            input: dict{offset, value}
            output: dict{offset, value} - 15s grid with nan values
        """
        grid_hrv = retime(self.hr_15s, 15, 86400, False)
                    
        filled_hrv = self.__inputation(grid_hrv)
        self.hrv_clean = moving_average(filled_hrv, 3)
        if(self.showPlots):
            showpreprocess(grid=grid_hrv, filled=filled_hrv,
                           avged=self.hrv_clean, title="HR preprocess", save=self.save)
        return self.hrv_clean

    def __inputation(self, grid_hr: dict) -> dict:
        [start, end, isle] = findNanIsland(grid_hr, 15, 120)
        return fillNanIslands(grid_hr, start, end)

    def segmenting(self, bbi, step):
        """support function to calculate sdnnaX and sdnniX

        Args:
            bbi (np.array): beat to beat intervals
            step (int): number of values to be considered in the segmentation

        Returns:
            np.array: sdnnaX and sdnniX for the specified step (SDNNAX: standard deviation of average of bbi, SDNNIX: average of standard deviation of bbi)
        """
        segments1 = []
        for i in np.arange(len(bbi), step=step):
            segments1.append(bbi[i:i+4])
        if len(segments1[-1]) != 4:
            segments1 = segments1[:len(segments1)-1]
        segments1 = np.array(segments1)[np.sum(
            np.where(np.isfinite(segments1), 1, 0), axis=1) > 0]
        return [np.nanstd(np.nanmean(segments1, axis=1)), np.nanmean(np.nanstd(segments1, axis=1))]

    # from NEUROKIT2 package

    def _hrv_nonlinear_poincare_hra(self, rri, out):
        """Heart Rate Asymmetry Indices.
        - Asymmetry of Poincaré plot (or termed as heart rate asymmetry, HRA) - Yan (2017)
        - Asymmetric properties of long-term and total heart rate variability - Piskorski (2011)
        """

        N = len(rri) - 1
        x = rri[:-1]  # rri_n, x-axis
        y = rri[1:]  # rri_plus, y-axis

        diff = y - x
        # set of points above IL where y > x
        decelerate_indices = np.where(diff > 0)[0]
        # set of points below IL where y < x
        accelerate_indices = np.where(diff < 0)[0]
        nochange_indices = np.where(diff == 0)[0]

        # Distances to centroid line l2
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        dist_l2_all = abs((x - centroid_x) + (y - centroid_y)) / np.sqrt(2)

        # Distances to LI
        dist_all = abs(y - x) / np.sqrt(2)

        # Calculate the angles
        # phase angle LI - phase angle of i-th point
        theta_all = abs(np.arctan(1) - np.arctan(y / x))
        # Calculate the radius
        r = np.sqrt(x ** 2 + y ** 2)
        # Sector areas
        S_all = 1 / 2 * theta_all * r ** 2

        # Guzik's Index (GI)
        den_GI = np.sum(dist_all)
        num_GI = np.sum(dist_all[decelerate_indices])
        out["GI"] = (num_GI / den_GI) * 100

        # Slope Index (SI)
        den_SI = np.sum(theta_all)
        num_SI = np.sum(theta_all[decelerate_indices])
        out["SI"] = (num_SI / den_SI) * 100

        # Area Index (AI)
        den_AI = np.sum(S_all)
        num_AI = np.sum(S_all[decelerate_indices])
        out["AI"] = (num_AI / den_AI) * 100

        # Porta's Index (PI)
        m = N - len(nochange_indices)  # all points except those on LI
        b = len(accelerate_indices)  # number of points below LI
        if(m != 0):
            out["PI"] = (b / m) * 100

        # Short-term asymmetry (SD1)
        sd1d = np.sqrt(np.sum(dist_all[decelerate_indices] ** 2) / (N - 1))
        sd1a = np.sqrt(np.sum(dist_all[accelerate_indices] ** 2) / (N - 1))

        sd1I = np.sqrt(sd1d ** 2 + sd1a ** 2)
        out["C1d"] = (sd1d / sd1I) ** 2
        out["C1a"] = (sd1a / sd1I) ** 2
        out["SD1d"] = sd1d  # SD1 deceleration
        out["SD1a"] = sd1a  # SD1 acceleration
        # out["SD1I"] = sd1I  # SD1 based on LI, whereas SD1 is based on centroid line l1

        # Long-term asymmetry (SD2)
        longterm_dec = np.sum(dist_l2_all[decelerate_indices] ** 2) / (N - 1)
        longterm_acc = np.sum(dist_l2_all[accelerate_indices] ** 2) / (N - 1)
        longterm_nodiff = np.sum(dist_l2_all[nochange_indices] ** 2) / (N - 1)

        sd2d = np.sqrt(longterm_dec + 0.5 * longterm_nodiff)
        sd2a = np.sqrt(longterm_acc + 0.5 * longterm_nodiff)

        sd2I = np.sqrt(sd2d ** 2 + sd2a ** 2)
        out["C2d"] = (sd2d / sd2I) ** 2
        out["C2a"] = (sd2a / sd2I) ** 2
        out["SD2d"] = sd2d  # SD2 deceleration
        out["SD2a"] = sd2a  # SD2 acceleration
        # out["SD2I"] = sd2I  # identical with SD2

        # Total asymmerty (SDNN)
        sdnnd = np.sqrt(0.5 * (sd1d ** 2 + sd2d ** 2))  # SDNN deceleration
        sdnna = np.sqrt(0.5 * (sd1a ** 2 + sd2a ** 2))  # SDNN acceleration
        # should be similar to sdnn in hrv_time
        sdnn = np.sqrt(sdnnd ** 2 + sdnna ** 2)
        out["Cd"] = (sdnnd / sdnn) ** 2
        out["Ca"] = (sdnna / sdnn) ** 2
        out["SDNNd"] = sdnnd
        out["SDNNa"] = sdnna

        return out

    def _hrv_nonlinear_fragmentation(self, rri, out):
        """Heart Rate Fragmentation Indices - Costa (2017)
        The more fragmented a time series is, the higher the PIP, IALS, PSS, and PAS indices will be.
        """

        diff_rri = np.diff(rri)
        zerocrossings = signal_zerocrossings(diff_rri)

        # Percentage of inflection points (PIP)
        out["PIP"] = len(zerocrossings) / len(rri)

        # Inverse of the average length of the acceleration/deceleration segments (IALS)
        accelerations = np.where(diff_rri > 0)[0]
        decelerations = np.where(diff_rri < 0)[0]
        consecutive = find_consecutive(
            accelerations) + find_consecutive(decelerations)
        lengths = [len(i) for i in consecutive]
        out["IALS"] = 1 / np.average(lengths)

        # Percentage of short segments (PSS) - The complement of the percentage of NN intervals in
        # acceleration and deceleration segments with three or more NN intervals
        out["PSS"] = np.sum(np.asarray(lengths) < 3) / len(lengths)

        # Percentage of NN intervals in alternation segments (PAS). An alternation segment is a sequence
        # of at least four NN intervals, for which heart rate acceleration changes sign every beat. We note
        # that PAS quantifies the amount of a particular sub-type of fragmentation (alternation). A time
        # series may be highly fragmented and have a small amount of alternation. However, all time series
        # with large amount of alternation are highly fragmented.
        alternations = find_consecutive(zerocrossings)
        lengths = [len(i) for i in alternations]
        out["PAS"] = np.sum(np.asarray(lengths) >= 4) / len(lengths)

        return out
