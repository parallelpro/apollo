#!/usr/bin/env/ python
# coding: utf-8

import numpy as np 
import matplotlib.pyplot as plt

__all__ = ['smooth_ps', 'smooth', 'smooth_series', 'auto_correlate', 'cross_correlate', 
           'lorentzian', 'spectral_response', 'echelle', 'plot_mcmc_traces', 'return_2dmap_axes']

def smooth_ps(freq, power, windowSize=0.25, windowType='flat',
                                samplingInterval=None):
    '''
    Return the moving average of a power spectrum, with a changing width of the window

    Input:
    freq: array-like[N,] in muHz
    power: array-like[N,]
    windowSize: float, in unit of the p-mode large separation
    windowType: flat/hanning/hamming/bartlett/blackman/gaussian
    samplingInterval: the time between adjacent sampling points.
    '''

    if len(freq) != len(power): 
        raise ValueError("freq and power must have equal size.")
        
    if samplingInterval is None: samplingInterval = np.median(freq[1:-1] - freq[0:-2])

    freqp = np.arange(np.min(freq),np.max(freq),samplingInterval)
    powerp = np.interp(freqp, freq, power)
    powersp = np.zeros(powerp.shape)
    
    numax = np.logspace(1,4,20)
    delta_nu = 0.263*numax**0.772 # stello+09 relation
    numax[0] = 0.
    numax = np.append(numax,np.inf)
    for iw in range(len(numax)-1):
        idx = (freqp>=numax[iw]) & (freqp<=numax[iw+1])
        window_len = int(delta_nu[iw]*windowSize/samplingInterval)
        if window_len % 2 == 0:
            window_len = window_len + 1
        if np.sum(idx) <= window_len:
            powersp[idx] = powerp[idx]
        else:
            inputArray = np.concatenate((np.ones(window_len)*powerp[idx][window_len:0:-1], powerp[idx], np.ones(window_len)*powerp[idx][-1:-window_len-1:-1]))
            powersp[idx] = smooth_series(inputArray, window_len, window=windowType)[window_len:window_len+np.sum(idx)]

    powers = np.interp(freq, freqp, powersp)
    return powers


def smooth(x, y, windowSize, windowType, samplingInterval=None):
    '''
    Wrapping a sliding-average smooth function.

    Input:
    x: the independent variable of the time series.
    y: the dependent variable of the time series.
    windowSize: the period/width of the sliding window.
    windowType: flat/hanning/hamming/bartlett/blackman/gaussian
    samplingInterval: the time between adjacent sampling points.

    Output:
    yf: the smoothed time series with the exact same points as x.

    '''

    if len(x) != len(y): 
        raise ValueError("x and y must have equal size.")
        
    if samplingInterval is None: samplingInterval = np.median(x[1:-1] - x[0:-2])

    xp = np.arange(np.min(x),np.max(x),samplingInterval)
    yp = np.interp(xp, x, y)
    window_len = int(windowSize/samplingInterval)
    if window_len % 2 == 0:
        window_len = window_len + 1
    ys = smooth_series(yp, window_len, window = windowType)
    yf = np.interp(x, xp, ys)

    return yf


def smooth_series(x, window_len = 11, window = "hanning"):
    # stole from https://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman", "gaussian"]:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = x #np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == "flat":
        w = np.ones(window_len,"d")
    elif window == "gaussian":
        w = gaussian(np.arange(-window_len*3, window_len*3,1), 
                    0, window_len, 1./(np.sqrt(2*np.pi)*window_len))
    else:
        w = eval("np."+window+"(window_len)") 
    
    y = np.convolve(w/w.sum(),s,mode="same")
    return y


def auto_correlate(x, y, ifInterpolate=True, samplingInterval=None): 

    '''
    Generate autocorrelation coefficient as a function of lag.

    Input:
        x: array-like[N,]
        y: array-like[N,]
        ifInterpolate: True or False. True is for unevenly spaced time series.
        samplinginterval: float. Default value: the median of intervals.

    Output:
        lag: time lag.
        rho: autocorrelation coeffecient.

    '''

    if len(x) != len(y): 
        raise ValueError("x and y must have equal size.")

    if ifInterpolate:
        samplingInterval = np.median(np.diff(x)) if (samplingInterval is None) else samplingInterval
        xp = np.arange(np.min(x),np.max(x),samplingInterval)
        yp = np.interp(xp, x, y)
        x, y = xp, yp

    new_y = y - np.mean(y)
    aco = np.correlate(new_y, new_y, mode='same')

    N = len(aco)
    lag = x[int(N/2):N] - x[int(N/2)]
    rho = aco[int(N/2):N] / np.var(y)
    rho = rho / np.max(rho)

    return lag, rho


def cross_correlate(x, y1, y2, ifInterpolate=True, samplingInterval=None): 
    '''
    Generate autocorrelation coefficient as a function of lag.

    Input:
        x: array-like[N,]
        y1: array-like[N,]
        y2: array-like[N,]
        ifInterpolate: True or False. True is for unevenly spaced time series.
        samplinginterval: float. Default value: the median of intervals.

    Output:
        lag: time lag.
        rho: autocorrelation coeffecient.

    '''

    if (len(x) != len(y1)) or (len(x) != len(y2)): 
        raise ValueError("x and y1 and y2 must have equal size.")

    if ifInterpolate:
        samplingInterval = np.median(x[1:-1] - x[0:-2]) if (samplingInterval is None) else samplingInterval
        xp = np.arange(np.min(x),np.max(x),samplingInterval)
        yp1 = np.interp(xp, x, y1)
        yp2 = np.interp(xp, x, y2)
        x, y1, y2 = xp, yp1, yp2

    new_y1 = y1 - np.mean(y1)
    new_y2 = y2 - np.mean(y2)
    aco = np.correlate(new_y1, new_y2, mode='same')

    N = len(aco)
    lag = x[int(N/2):N] - x[int(N/2)]
    rho = aco[int(N/2):N] / (np.std(yp1)*np.std(yp2))
    rho = rho / np.max(rho)

    return lag, rho



def lorentzian(x, mu, gamma, height):
    '''
    Return the value of lorentzian given parameters.

    Input:
    x: array-like[N,]
    mu, gamma, height: float

    Output:
    y: the dependent variable of the time series.

    '''
    return height / (1 + (x-mu)**2.0/gamma**2.0)


def spectral_response(x, fnyq):
	sincfunctionarg = (np.pi/2.0)*x/fnyq
	response = (np.sin(sincfunctionarg)/sincfunctionarg)**2.0
	return response


def echelle(freq, ps, Dnu, fmin=None, fmax=None, echelletype="single", offset=0.0):
    '''
    Make an echelle plot used in asteroseismology.
    
    Input parameters
    ----
    freq: 1d array-like, freq
    ps: 1d array-like, power spectrum
    Dnu: float, length of each vertical stack (Dnu in a frequency echelle)
    fmin: float, minimum frequency to be plotted
    fmax: float, maximum frequency to be plotted
    echelletype: str, `single` or `replicated`
    offset: float, an amount by which the diagram is shifted horizontally
    
    Return
    ----
    z: a 2d numpy.array, folded power spectrum
    extent: a list, edges (left, right, bottom, top) 
    x: a 1d numpy.array, horizontal axis
    y: a 1d numpy.array, vertical axis
    
    Users can create an echelle diagram with the following command:
    ----
    
    import matplotlib.pyplot as plt
    z, ext = echelle(freq, power, Dnu, fmin=numax-4*Dnu, fmax=numax+4*Dnu)
    plt.imshow(z, extent=ext, aspect='auto', interpolation='nearest')
    
    '''
    
    if fmin is None: fmin=0.
    if fmax is None: fmax=np.nanmax(x)

    fmin -= offset
    fmax -= offset
    freq -= offset

    fmin = 1e-4 if fmin<Dnu else fmin - (fmin % Dnu)

    # define plotting elements
    resolution = np.median(np.diff(freq))
    # number of vertical stacks
    n_stack = int((fmax-fmin)/Dnu) 
    # number of point per stack
    n_element = int(Dnu/resolution) 

    fstart = fmin - (fmin % Dnu)
    
    z = np.zeros([n_stack, n_element])
    base = np.linspace(0, Dnu, n_element) if echelletype=='single' else np.linspace(0, 2*Dnu, n_element)
    for istack in range(n_stack):
        z[-istack-1,:] = np.interp(fstart+istack*Dnu+base, freq, ps)
    
    extent = (0, Dnu, fstart, fstart+n_stack*Dnu) if echelletype=='single' else (0, 2*Dnu, fstart, fstart+n_stack*Dnu)
    
    x = base
    y = fstart + np.arange(0, n_stack+1, 1)*Dnu + Dnu/2
    
    return z, extent, x, y



def return_2dmap_axes(NSquareBlocks):

    # Some magic numbers for pretty axis layout.
    # stole from corner
    Kx = int(np.ceil(NSquareBlocks**0.5))
    Ky = Kx if (Kx**2-NSquareBlocks) < Kx else Kx-1

    factor = 2.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin, default=0.2
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.30         # w/hspace size
    plotdimx = factor * Kx + factor * (Kx - 1.) * whspace
    plotdimy = factor * Ky + factor * (Ky - 1.) * whspace
    dimx = lbdim + plotdimx + trdim
    dimy = lbdim + plotdimy + trdim

    # Create a new figure if one wasn't provided.
    fig, axes = plt.subplots(Ky, Kx, figsize=(dimx, dimy), squeeze=False)

    # Format the figure.
    l = lbdim / dimx
    b = lbdim / dimy
    t = (lbdim + plotdimy) / dimy
    r = (lbdim + plotdimx) / dimx
    fig.subplots_adjust(left=l, bottom=b, right=r, top=t,
                        wspace=whspace, hspace=whspace)
    axes = np.concatenate(axes)

    return fig, axes

def plot_mcmc_traces(Ndim, samples, paramsNames):

    fig, axes = return_2dmap_axes(Ndim)

    for i in range(Ndim):
        ax = axes[i]
        evol = samples[:,i]
        Npoints = samples.shape[0]
        ax.plot(np.arange(Npoints)/Npoints, evol, color="gray", lw=1, zorder=1)
        Nseries = int(len(evol)/15.0)
        evol_median = np.array([np.median(evol[i*Nseries:(i+1)*Nseries]) for i in range(0,15)])
        evol_std = np.array([np.std(evol[i*Nseries:(i+1)*Nseries]) for i in range(0,15)])
        evol_x = np.array([np.median(np.arange(Npoints)[i*Nseries:(i+1)*Nseries]/Npoints) for i in range(0,15)])
        ax.errorbar(evol_x, evol_median, yerr=evol_std, color="C0", ecolor="C0", capsize=2)
        ax.set_ylabel(paramsNames[i])

    for ax in axes[i+1:]:
        fig.delaxes(ax)

    return fig