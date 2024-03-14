from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import numpy as np
import itertools

def findNanIsland(data: dict, sampleTime: int, maxGap: int) -> list[np.array]:
    """ find NAN island in signal

    Args:
        data (dict): data input with time offset as key and data as value
        sampleTime (int): sample time of data
        maxGap (int): maximum allowed number of consecutive nan

    Returns:
        list[np.array]: list of three arrays: start index, end index, isle mask
    """
    nancount = np.zeros(len(data))
    firstval = False
    lastval = len(data) - next(x for x,
                               val in enumerate(np.flipud(list(data.values()))) if not np.isnan(val))
    for i, val in enumerate(data.values()):
        if(i == lastval):
            break
        if(np.isnan(val)):
            if(firstval):
                nancount[i] = nancount[i-1] + 1
        else:
            firstval = True
    isle = np.full(len(nancount), False)  # nancount.copy()
    notIsle = False
    start = []
    end = []
    i = 0
    flipped = np.flip(nancount)
    while i < len(flipped):
        # if(val == 0):
        #     if(isle[-i+1] != 0):
        #         start.append(len(nancount)-i)
        #     isle[-i] = 0
        #     notIsle = False
        # if(val > round(maxGap/sampleTime)):
        #     notIsle = True
        # if(not notIsle):
        #     if(val > 0):
        #         isle[-i] = 1
        #         if(isle[-i+1] == 0):
        #             end.append(len(nancount)-i-1)
        # else:
        #     isle[-i] = 0
        val = flipped[i]
        if(val <= round(maxGap/sampleTime) and val > 0):
            end.append(len(nancount)-i-1)
            j = i
            while flipped[j] != 0:
                j = j+1
                isle[-j] = True
            i = j
            start.append(len(nancount)-i)
        elif(val > round(maxGap/sampleTime)):
            j = i
            while flipped[j] != 0:
                j = j+1
            i = j
        i = i+1

    start = np.flipud(start)
    end = np.flipud(end)
    # isle = [bool(i) for i in isle]
    return [start, end, isle]


def fillNanIslands(data: dict, start: np.array, end: np.array) -> dict:
    """ fill nan islands with interpolated data

    Args:
        data (dict): inpuut data
        start (np.array): array of start indexes of nan islands
        end (np.array): array of end indexes of nan islands

    Returns:
        dict: output interpolated data
    """
    d = data.copy()
    for i, v in enumerate(start):
        for n in range(v, end[i]+1):
            k = list(d.keys())[n]
            d[k] = np.interp(
                v, [v-1, end[i]+1], [list(d.values())[v-1], list(d.values())[end[i]+1]])

    return d


def retime(data: dict, step: int, maxTime: int, cumulative: bool = False) -> dict:
    """ function to retime data on uniform grid

    Args:
        data (dict): input data
        step (int): step of grid
        maxTime (int): maximum grid time
        cumulative (bool, optional): if the data is cumulative, e.g. steps during the day. Defaults to False.

    Returns:
        dict: retimed data
    """
    grid = {}
    for i in np.arange(0, maxTime, step):
        grid[i] = np.nan
    count = dict.fromkeys(np.arange(0, maxTime, step), np.nan)
    times = np.fromiter(grid.keys(), dtype=float)
    for t in data:
        distances = abs(t-times)
        nearest = times[np.where(distances == (min(distances)))[0][0]]
        if(np.isnan((grid[nearest]))):
            grid[nearest] = data[t]
            count[nearest] = 1
        else:
            grid[nearest] += data[t]
            count[nearest] += 1
    if(not cumulative):
        for k in grid:
            grid[k] = grid[k]/count[k]
    return grid

def moving_average(data: dict, n=3, roundValues=False):
    """moving average

    Args:
        data (dict): input data
        n (int, optional): number of elements in the moving window. Defaults to 3.

    Returns:
        dict: averaged data
    """
    values = np.fromiter(data.values(), dtype=float)
    a = np.ma.masked_array(values, np.isnan(values))
    ret = np.cumsum(a.filled(0))
    ret[n:] = ret[n:] - ret[:-n]
    counts = np.cumsum(~a.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ret[~a.mask] /= counts[~a.mask]
    if(roundValues):
        ret[~a.mask] = np.around(ret[~a.mask])
    ret[a.mask] = np.nan
    return dict(zip(list(data.keys()), ret))


def polifit_nan(data: dict, deg: int = 1):
    """polyfit wrapper for nan values

    Args:
        data (dict): input data
        deg (int, optional): Degree of fitting function. Defaults to 1.

    Returns:
        as numpy's polyfit function
    """
    x = np.fromiter(data.keys(), dtype=float)
    y = np.fromiter(data.values(), dtype=float)
    idx = np.isfinite(x) & np.isfinite(y)
    if(len(y[idx])>0):
        return np.polyfit(x[idx], y[idx], deg)
    else:
        return [None]


def r2(yhat: np.array, y: np.array):
    """support function to calculate R^2 value

    Args:
        yhat (np.array): fitted values
        y (np.array): original values

    Returns:
       r2 (float): r2 value
    """
    idx = np.isfinite(y)
    y = y[idx]
    yhat = yhat[idx]
    ybar = sum(y)/len(y)
    SST = sum((y - ybar)**2)
    SSreg = sum((yhat - ybar)**2)
    r2 = SSreg/SST
    return r2


# copied from NEUROKIT2 package

def find_consecutive(x):
    """Find and group consecutive values in a list.
    Parameters
    ----------
    x : list
        The list to look in.
    Returns
    -------
    list
        A list of tuples corresponding to groups containing all the consecutive numbers.
    """

    return [tuple(g) for k, g in itertools.groupby(x, lambda n, c=itertools.count(): n - next(c))]


def signal_zerocrossings(signal, direction="both"):
    """Locate the indices where the signal crosses zero.
    Note that when the signal crosses zero between two points, the first index is returned.
    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    direction : str
        Direction in which the signal crosses zero, can be "positive", "negative" or "both" (default).
    Returns
    -------
    array
        Vector containing the indices of zero crossings.
    """
    df = np.diff(np.sign(signal))
    if direction in ["positive", "up"]:
        zerocrossings = np.where(df > 0)[0]
    elif direction in ["negative", "down"]:
        zerocrossings = np.where(df < 0)[0]
    else:
        zerocrossings = np.nonzero(np.abs(df) > 0)[0]

    return zerocrossings


def fractal_dfa(signal, windows="default", overlap=True, integrate=True,
                order=1, multifractal=False, q=2, show=False):
    """(Multifractal) Detrended Fluctuation Analysis (DFA or MFDFA).

    Python implementation of Detrended Fluctuation Analysis (DFA) or
    Multifractal DFA of a signal. Detrended fluctuation analysis, much like the
    Hurst exponent, is used to find long-term statistical dependencies in time
    series.

    This function can be called either via `fractal_dfa()` or
    `complexity_dfa()`, and its multifractal variant can be directly accessed
    via `fractal_mfdfa()` or `complexity_mfdfa()`.

    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.

    windows : list
        A list containing the lengths of the windows (number of data points in
        each subseries). Also referred to as 'lag' or 'scale'. If 'default',
        will set it to a logarithmic scale (so that each window scale has the
        same weight) with a minimum of 4 and maximum of a tenth of the length
        (to have more than 10 windows to calculate the average fluctuation).

    overlap : bool
        Defaults to True, where the windows will have a 50% overlap
        with each other, otherwise non-overlapping windows will be used.

    integrate : bool
        It is common practice to convert the signal to a random walk (i.e.,
        detrend and integrate, which corresponds to the signal 'profile'). Note
        that it leads to the flattening of the signal, which can lead to the
        loss of some details (see Ihlen, 2012 for an explanation). Note that for
        strongly anticorrelated signals, this transformation should be applied
        two times (i.e., provide `np.cumsum(signal - np.mean(signal))` instead
        of `signal`).

    order : int

       The order of the polynomial trend for detrending, 1 for the linear trend.

    multifractal : bool
        If true, compute Multifractal Detrended Fluctuation Analysis (MFDFA), in
        which case the argument `q` is taken into account.

    q : list or np.array (default `2`)
        The sequence of fractal exponents when `multifractal=True`. Must be a
        sequence between `-10` and `10` (note that zero will be removed, since
        the code does not converge there). Setting `q = 2` (default) gives a
        result of a standard DFA. For instance, Ihlen (2012) uses
        `q = [-5, -3, -1, 0, 1, 3, 5]`. In general, positive q moments amplify
        the contribution of fractal components with larger amplitude and
        negative q moments amplify the contribution of fractal with smaller
        amplitude (Kantelhardt et al., 2002)

    show : bool
        Visualise the trend between the window size and the fluctuations.

    Returns
    ----------
    dfa : dict
        If `multifractal` is False, the dictionary contains q value, a series of windows, fluctuations of each window and the
        slopes value of the log2(windows) versus log2(fluctuations) plot. If `multifractal` is True, the dictionary
        additionally contains the parameters of the singularity spectrum (scaling exponents, singularity dimension, singularity
        strength; see `singularity_spectrum()` for more information).


    References
    -----------
    - Ihlen, E. A. F. E. (2012). Introduction to multifractal detrended
      fluctuation analysis in Matlab. Frontiers in physiology, 3, 141.

    - Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S.,
      Bunde, A., & Stanley, H. E. (2002). Multifractal detrended fluctuation
      analysis of nonstationary time series. Physica A: Statistical
      Mechanics and its Applications, 316(1-4), 87-114.

    - Hardstone, R., Poil, S. S., Schiavone, G., Jansen, R., Nikulin, V. V.,
      Mansvelder, H. D., & Linkenkaer-Hansen, K. (2012). Detrended
      fluctuation analysis: a scale-free view on neuronal oscillations.
      Frontiers in physiology, 3, 450.

    - `nolds <https://github.com/CSchoel/nolds/>`_

    - `MFDFA <https://github.com/LRydin/MFDFA/>`_

    - `Youtube introduction <https://www.youtube.com/watch?v=o0LndP2OlUI>`_

    """
    # Sanity checks
    n = len(signal)
    windows = _fractal_dfa_findwindows(n, windows)
    # Return warning for too short windows
    _fractal_dfa_findwindows_warning(windows, n)

    # Preprocessing
    if integrate is True:
        signal = np.cumsum(signal - np.mean(signal))  # Get signal profile

    # Sanitize fractal power (cannot be close to 0)
    q = _cleanse_q(q, multifractal=multifractal)

    # obtain the windows and fluctuations
    windows, fluctuations = _fractal_dfa(signal=signal,
                                         windows=windows,
                                         overlap=overlap,
                                         integrate=integrate,
                                         order=order,
                                         multifractal=multifractal,
                                         q=q
                                         )

    if len(fluctuations) == 0:
        return np.nan

    slopes = _slopes(windows, fluctuations, q)
    out = {'q': q[:, 0],
           'windows': windows,
           'fluctuations': fluctuations,
           'slopes': slopes}

    if multifractal is True:
        singularity = singularity_spectrum(windows=windows,
                                           fluctuations=fluctuations,
                                           q=q,
                                           slopes=slopes)
        out.update(singularity)

    return out

# =============================================================================
# Utilities
# =============================================================================


def _fractal_dfa(signal, windows="default", overlap=True, integrate=True,
                 order=1, multifractal=False, q=2):
    """Does the heavy lifting for `fractal_dfa()`.

    Returns
    ----------
    windows : list
        A list containing the lengths of the windows

    fluctuations : np.ndarray
        The detrended fluctuations, from DFA or MFDFA.
    """

    # Function to store fluctuations. For DFA this is an array. For MFDFA, this
    # is a matrix of size (len(windows),len(q))
    n = len(signal)
    fluctuations = np.zeros((len(windows), len(q)))

    # Start looping over windows
    for i, window in enumerate(windows):

        # Get window
        segments = _fractal_dfa_getwindow(signal, n, window, overlap=overlap)

        # Get polynomial trends
        trends = _fractal_dfa_trends(segments, window, order=order)

        # Get local fluctuation
        fluctuations[i] = _fractal_dfa_fluctuation(segments,
                                                   trends,
                                                   multifractal,
                                                   q
                                                   )

    # I would not advise this part. I understand the need to remove zeros, but I
    # would instead suggest masking it with numpy.ma masked arrays. Note that
    # when 'q' is a list,  windows[nonzero] increases in size.

    # Filter zeros
    # nonzero = np.nonzero(fluctuations)[0]
    # windows = windows[nonzero]
    # fluctuations = fluctuations[nonzero]

    return windows, fluctuations


# =============================================================================
#  Utils MFDFA
# =============================================================================
# This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141
#
# It was designed by Leonardo Rydin Gorjão as part of MFDFA
# (https://github.com/LRydin/MFDFA). It is included here by the author and
# altered to fit NK to the best of its extent.

def singularity_spectrum(windows, fluctuations, q, slopes):
    """Extract the slopes of the fluctuation function to futher obtain the
    singularity strength `α` (or Hölder exponents) and singularity spectrum
    `f(α)` (or fractal dimension). This is iconically shaped as an inverse
    parabola, but most often it is difficult to obtain the negative `q` terms,
    and one can focus on the left side of the parabola (`q>0`).

    Note that these measures rarely match the theoretical expectation,
    thus a variation of ± 0.25 is absolutely reasonable.

    The parameters are mostly identical to `fractal_mfdfa()`, as one needs to
    perform MFDFA to obtain the singularity spectrum. Calculating only the
    DFA is insufficient, as it only has `q=2`, and a set of `q` values are
    needed. Here defaulted to `q = list(range(-5,5))`, where the `0` element
    is removed by `_cleanse_q()`.

    Parameters
    ----------
    windows : list
        A list containing the lengths of the windows. Output of `_fractal_dfa()`.

    fluctuations : np.ndarray
        The detrended fluctuations, from DFA or MFDFA. Output of `_fractal_dfa()`.

    q : list or np.array (default `np.linspace(-10,10,41)`)
        The sequence of fractal exponents. Must be a sequence between -10
        and 10 (note that zero will be removed, since the code does not converge
        there). If "default", will takes the form `np.linspace(-10,10,41)`.

    slopes : np.ndarray
        The slopes of each `q` power obtained with MFDFA. Output of `_slopes()`.

    Returns
    -------
    tau: np.array
        Scaling exponents `τ`. A usually increasing function of `q` from
        which the fractality of the data can be determined by its shape. A truly
        linear tau indicates monofractality, whereas a curved one (usually
        curving around small `q` values) indicates multifractality.

    hq: np.array
        Singularity strength `hq`. The width of this function indicates the
        strength of the multifractality. A width of `max(hq) - min(hq) ≈ 0`
        means the data is monofractal.

    Dq: np.array
        Singularity spectrum `Dq`. The location of the maximum of `Dq` (with
         `hq` as the abscissa) should be 1 and indicates the most prominent
         exponent in the data.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.
    """

    # Calculate τ
    tau = q[:, 0] * slopes - 1

    # Calculate hq or α, which needs tau
    hq = np.gradient(tau) / np.gradient(q[:, 0])

    # Calculate Dq or f(α), which needs tau and q
    Dq = q[:, 0] * hq - tau

    # Calculate the singularity
    ExpRange = np.max(hq) - np.min(hq)
    ExpMean = np.mean(hq)
    DimRange = np.max(Dq) - np.min(Dq)
    DimMean = np.mean(Dq)
    out = {'tau': tau,
           'hq': hq,
           'Dq': Dq,
           'ExpRange': ExpRange,
           'ExpMean': ExpMean,
           'DimRange': DimRange,
           'DimMean': DimMean}

    return out


# =============================================================================
#  Utils
# =============================================================================

def _cleanse_q(q=2, multifractal=False):
    # TODO: Add log calculator for q ≈ 0

    # Enforce DFA in case 'multifractal = False' but 'q' is not 2
    if multifractal is False:
        q = 2
    else:
        if isinstance(q, int):
            q = [-5, -3, -1, 0, 1, 3, 5]

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=float)

    # Ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1
    q = q[(q < -0.1) + (q > 0.1)]

    # Reshape q to perform np.float_power
    q = q.reshape(-1, 1)

    return q


def _slopes(windows, fluctuations, q):
    """
    Extract the slopes of each `q` power obtained with MFDFA to later produce
    either the singularity spectrum or the multifractal exponents.

    Notes
    -----
    This was first designed and implemented by Leonardo Rydin in
    `MFDFA <https://github.com/LRydin/MFDFA/>`_ and ported here by the same.

    """

    # Ensure mfdfa has the same q-power entries as q
    if fluctuations.shape[1] != q.shape[0]:
        raise ValueError(
            "Fluctuation function and q powers don't match in dimension.")

    # Allocated array for slopes
    slopes = np.zeros(len(q))
    # Find slopes of each q-power
    for i in range(len(q)):
        old_setting = np.seterr(divide="ignore", invalid="ignore")
        slopes[i] = np.polyfit(
            np.log2(windows), np.log2(fluctuations[:, i]), 1)[0]
        np.seterr(**old_setting)

    return slopes


def _fractal_dfa_findwindows(n, windows="default"):
    # Convert to array
    if isinstance(windows, list):
        windows = np.asarray(windows)

    # Default windows number
    if windows is None or isinstance(windows, str):
        windows = int(n / 10)

    # Default windows sequence
    if isinstance(windows, int):
        windows = np.unique(windows)  # keep only unique

    return windows


def _fractal_dfa_findwindows_warning(windows, n):

    # Check windows
    if len(windows) < 2:
        raise ValueError(
            "NeuroKit error: fractal_dfa(): more than one window is needed."
        )

    if np.min(windows) < 2:
        raise ValueError(
            "NeuroKit error: fractal_dfa(): there must be at least 2 data "
            "points in each window"
        )
    if np.max(windows) >= n:
        raise ValueError(
            "NeuroKit error: fractal_dfa(): the window cannot contain more data"
            " points than the" "time series."
        )


def _fractal_dfa_getwindow(signal, n, window, overlap=True):
    # This function reshapes the segments from a one-dimensional array to a
    # matrix for faster polynomail fittings. 'Overlap' reshapes into several
    # overlapping partitions of the data

    if overlap:
        segments = np.array([signal[i: i + window]
                             for i in np.arange(0, n - window, window // 2)
                             ])
    else:
        segments = signal[: n - (n % window)]
        segments = segments.reshape((signal.shape[0] // window, window))

    return segments


def _fractal_dfa_trends(segments, window, order=1):
    x = np.arange(window)

    coefs = np.polyfit(x[:window], segments.T, order).T

    # TODO: Could this be optimized? Something like np.polyval(x[:window],coefs)
    trends = np.array([np.polyval(coefs[j], x)
                       for j in np.arange(len(segments))
                       ])

    return trends


def _fractal_dfa_fluctuation(segments, trends, multifractal=False, q=2):

    detrended = segments - trends

    if multifractal is True:
        var = np.var(detrended, axis=1)
        # obtain the fluctuation function, which is a function of the windows
        # and of q
        # ignore division by 0 warning
        old_setting = np.seterr(divide="ignore", invalid="ignore")
        pow = np.float_power(var, q / 2)
        for i in np.arange(0, len(pow)):
            pow[i] = [k if(np.isfinite(k)) else np.nan for k in pow[i]]
        fluctuation = \
            np.float_power(np.nanmean(pow, axis=1), 1 / q.T)
        np.seterr(**old_setting)

    else:
        # Compute Root Mean Square (RMS)
        fluctuation = np.sum(detrended ** 2, axis=1) / detrended.shape[1]
        fluctuation = np.sqrt(np.sum(fluctuation) / len(fluctuation))

    return fluctuation


def graph(data: dict, title: str, start: int = 0, save: bool = False):
    times = np.fromiter(data.keys(), dtype=float)
    starttime = datetime(2000, 1, 1).__add__(timedelta(seconds=start))
    end = np.fromiter(data.keys(), dtype=float)[-1]
    endtime = starttime.__add__(timedelta(seconds=end))
    timings = [starttime.__add__(timedelta(seconds=x)) for x in times]
    ti = np.linspace(0, len(timings)-1, 5, dtype=int)
    ticks = [timings[i] for i in ti]
    # ticks = [starttime, timings[len(timings)//3],
    #          timings[2*len(timings)//3], endtime]
    labels = [x.strftime("%H:%M") for x in ticks]
    fig = plt.figure(dpi=300, figsize=(6, 4))
    plt.plot(timings, np.fromiter(data.values(), dtype=float))
    plt.title(title)
    plt.xticks(ticks=ticks, labels=labels)
    plt.show()
    if save:
        fig.savefig('figures/'+title+'.png')


def showpreprocess(grid: dict, filled: dict, avged: dict, title: str, start: int = 0, save: bool = False):
    times = np.fromiter(grid.keys(), dtype=float)
    starttime = datetime(2000, 1, 1).__sub__(timedelta(seconds=start))
    end = np.fromiter(grid.keys(), dtype=float)[-1]
    endtime = starttime.__add__(timedelta(seconds=end))
    timings = [starttime.__add__(timedelta(seconds=x)) for x in times]
    ti = np.linspace(0, len(timings)-1, 5, dtype=int)
    ticks = [timings[i] for i in ti]
    ''' ticks = [starttime, timings[len(timings)//3],
             timings[2*len(timings)//3], endtime] '''
    labels = [x.strftime("%H:%M") for x in ticks]
    set1 = set(filled.items())
    set2 = set(grid.items())
    fill = set1 - set2
    filled_times = [starttime.__add__(
        timedelta(seconds=float(t[0]))) for t in list(fill)]
    filled_vals = [t[1] for t in list(fill)]
    fig = plt.figure(dpi=300, figsize=(6, 4))
    plt.plot(timings,
             np.fromiter(filled.values(), dtype=float), 'b:', label="Retimed data")
    plt.plot(filled_times,
             filled_vals, 'g*', label="Filled gaps")
    # plt.plot(timings,
    #          np.fromiter(grid.values(), dtype=float), 'b', label="grid")
    plt.plot(timings,
             np.fromiter(avged.values(), dtype=float), 'r', label="Averaged data")
    plt.title(title)
    plt.xticks(ticks=ticks, labels=labels)
    plt.legend()
    plt.show()
    if save:
        fig.savefig('figures/'+title+'.png')
