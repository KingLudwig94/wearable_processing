from tracemalloc import start
import numpy as np
from brainteaser_preprocessing.utils import fillNanIslands, findNanIsland, fractal_dfa, moving_average, retime, graph, showpreprocess
import hrvanalysis as hv
import antropy as ent


class RSP:
    """
    Class for respiration rate feature extraction
    """

    rsp_60s: dict
    sleep_duration: int
    showPlot: bool
    start_offset: int
    save: bool

    def __init__(self, rsp_60s, sleep_duration, showPlot=False, start_offset: int = 0, save=False):
        self.rsp_60s = rsp_60s
        self.sleep_duration = sleep_duration
        self.showPlot = showPlot
        self.start_offset = start_offset
        self.save = save

    def run(self):
        self.preprocess()
        if(self.showPlot):
            graph(self.rsp_clean, "Respiration", save=self.save)
        return self.compute()

    def compute(self) -> dict:
        # print('----RSP----')

        data = np.fromiter(self.rsp_clean.values(), dtype=float)
        datanan = data[np.isfinite(data)]
        mean = np.nanmean(data)
        std = np.nanstd(data)

        # variance
        summed = 0
        for x in datanan:
            summed += (x-mean)**2
        var = summed/(len(datanan))

        ap_ent = ent.app_entropy(datanan)
        samp_entropy = ent.sample_entropy(datanan)

        # bbi
        bbi = np.fromiter(self.bbi.values(), dtype=float)
        bbinan = bbi[np.isfinite(bbi)]
        bbidiff = np.diff(bbi)
        bbidiffnan = bbidiff[np.isfinite(bbidiff)]
        time_domain_features = hv.get_time_domain_features(bbinan)
        poincare_features = hv.get_poincare_plot_features(bbinan)
        freq_domain_features = hv.get_frequency_domain_features(
            bbinan, sampling_frequency=1/60)

        out = {'mean': mean, 'var': var, 'sdbb': std, 'sdsd': time_domain_features['sdsd'], 'rmssd': time_domain_features['rmssd'],
               'sd1': poincare_features['sd1'], 'sd2': poincare_features['sd2'], 'sd1sd2': poincare_features['ratio_sd2_sd1'], 'ApEn': ap_ent, 'SampEn': samp_entropy, }

        # copied from NEUROKIT2 package
        if len(bbinan) / 10 > 16:
            out["DFA_alpha1"] = fractal_dfa(bbinan, windows=np.arange(
                4, 17), multifractal=False)['slopes'][0]
            # For multifractal
            mdfa_alpha1 = fractal_dfa(bbinan,
                                      multifractal=True,
                                      q=np.arange(-5, 6),
                                      windows=np.arange(4, 17))

            out["DFA_alpha1_ExpRange"] = mdfa_alpha1['ExpRange']
            out["DFA_alpha1_ExpMean"] = mdfa_alpha1['ExpMean']
            out["DFA_alpha1_DimRange"] = mdfa_alpha1['DimRange']
            out["DFA_alpha1_DimMean"] = mdfa_alpha1['DimMean']
        if len(bbinan) > 65:
            out["DFA_alpha2"] = fractal_dfa(bbinan, windows=np.arange(
                16, 65), multifractal=False)['slopes'][0]
            # For multifractal
            mdfa_alpha2 = fractal_dfa(bbinan,
                                      multifractal=True,
                                      q=np.arange(-5, 6),
                                      windows=np.arange(16, 65))

            out["DFA_alpha2_ExpRange"] = mdfa_alpha2['ExpRange']
            out["DFA_alpha2_ExpMean"] = mdfa_alpha2['ExpMean']
            out["DFA_alpha2_DimRange"] = mdfa_alpha2['DimRange']
            out["DFA_alpha2_DimMean"] = mdfa_alpha2['DimMean']

        return out

    # #root mean square
    # def RR_RMS(self):
    #     data = self.bbi
    #     ps = np.abs (np.fft.fft (data)) ** 2
    #     rms_flat = np.sqrt(np.mean(np.absolute(self.bbi)**2))
    #     rms_fft = rms_flat(ps)/np.sqrt(len(ps))
    #     return rms_fft

    # #power spectrum
    # def power_spectrum(self):
    #     data = self.rsp_clean
    #     ps = np.abs(np.fft.fft(data))**2
    #     return ps
    # #power density for very low frequencies

    # #power density fro very high frequencies

    # #low frequency power/ total power =normaized low freq

    # # normalized high freq : high freq power/ total power

    # #maximum frequency
    # def max_freq(self):
    #     data=self.rsp_clean
    #     time_step=1/60
    #     freqs=np.fft.fftfreq(data.size, time_step)
    #     max_f=max(freqs)
    #     return max_f

    # #minimum frequency
    # def min_freq(self):
    #     data=self.rsp_clean
    #     time_step=1/60
    #     freqs = np.fft.fftfreq (data.size, time_step)
    #     min_f=min(freqs)
    #     return min_f

#     #approximate entropy
#     def rsp_ApEn(self):
#         data = self.rsp_clean
#         ap_ent = ent.app_entropy(data)
#         return ap_ent


#     # sample entropy
#     def entropy_rsp(self):
#         """"
#             Return the Shannon Entropy of the data sample.
#         """
#         data = self.rsp_clean
#         ent = 0.0
#         for freq in data:
#             ent += freq*np.log2(freq)
#         ent = -ent
#         return ent

# #alternative
#     def rsp_SampEnt(self):
#         data = self.rsp_clean
#         samp_entropy = ent.sample_entropy(data)
#         return samp_entropy

#     #singular spectrum analysis
#     def ssa_rsp(self):
#         data = self.rsp_clean
#         ssa = SingularSpectrumAnalysis(window_size=15)
#         ssa_new = ssa.transform(data)
#         return ssa_new

#     # fluctuation value generated from Detrended Fluctuation analysis
#     def dfa_rsp(self):
#         data = self.resp_clean
#         dfa = nolds.dfa(data)
#         return dfa

#     # long term fluctuation value
#     # computed only if there are more than 640 breath cycles in the signal

#     def RR_dfa(self):
#         data = self.rsp_clean
#         dfa = fathon.dfa(data)
#         return dfa

#     # multifractal DFA
#     def RR_mfdfa(self):
#         data = self.rsp_clean
#         mfdfa = MFDFA.MFDFA(data)
#         return mfdfa

    def preprocess(self) -> np.array:
        """
            Preprocessing of rsp
            input: dict{offset, value}
            output: dict{offset, value} - 60s grid with nan values
        """
        grid_rsp = retime(self.rsp_60s, 60, self.sleep_duration, False)
        filled_rsp = self.__inputation(grid_rsp)
        self.rsp_clean = moving_average(filled_rsp, 10, True)
        self.bbi = {k: 60/v for k, v in self.rsp_clean.items()}

        if(self.showPlot):
            showpreprocess(grid=grid_rsp, filled=filled_rsp,
                           avged=self.rsp_clean, title="Respiration preprocess",  start=self.start_offset, save=self.save)
        return self.rsp_clean

    def __inputation(self, grid_rsp: dict) -> dict:
        [start, end, isle] = findNanIsland(grid_rsp, 60, 60*10)
        return fillNanIslands(grid_rsp, start, end)
