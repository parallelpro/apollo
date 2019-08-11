import numpy as np
import matplotlib.pyplot as plt
import os


# customized
from scipy.optimize import minimize

class SolarlikeFourierAnalysis:
    """docstring for SolarlikeFourierAnalysis"""
    def __init__(self, starname, outputdir, fnyq, numax):
        sep = "\\" if os.name=="nt" else "/"
        self.starname = starname
        self.outputdir = outputdir  # "with a / in the end"
        assert outputdir.endswith(sep), "outputdir should end with "+sep
        self.fnyq = fnyq  # in microHz (muHz) 
        self.numax = numax  # in microHz (muHz)
        self.dnu = (self.numax/3050)**0.77 * 135.1 # Stello+2009 

        return

    # def pass_light_curves
    
    def pass_power_spectrum(self, freq, power, trimUpperLimitInDnu=None, 
        trimLowerLimitInDnu=None, ifGlobalMode=True):       
        idx = np.array(np.zeros(len(freq))+1, dtype=bool)
        freq = np.array(freq)
        power = np.array(power)
        if not trimUpperLimitInDnu is None:
            idx = (idx) & (freq<=self.numax+trimUpperLimitInDnu*self.dnu)
        if not trimLowerLimitInDnu is None:
            idx = (idx) & (freq>=self.numax-trimLowerLimitInDnu*self.dnu)

        if ifGlobalMode:
            self.freq = freq[idx]
            self.power = power[idx]
            return
        else:
            return freq[idx], power[idx]


    def __smooth_power(self, period=None):
        if period is None: period = self._dnu0/15.0 # microHz
        self.powers = self._smooth_wrapper(self.freq, self.power, period, "bartlett")
        return

    def _smooth_wrapper(x, y, period, windowtype, samplinginterval=None):

        if samplinginterval is None: samplinginterval = np.median(x[1:-1] - x[0:-2])
        if not windowtype in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        xp = np.arange(np.min(x), np.max(x), samplinginterval)
        yp = np.interp(xp, x, y)
        window_len = int(period/samplinginterval)
        if window_len % 2 == 0:
            window_len = window_len + 1
        if windowtype == "flat":
            w = np.ones(window_len,"d")
        else:
            w = eval("np."+windowtype+"(window_len)") 
        
        ys = np.convolve(w/w.sum(),yp,mode="same")
        yf = np.interp(x, xp, ys)

        return yf

    def __background_model(self, bgpara, bgtype="withgaussian", granNumber=1):
        flatNoiseLevel = bgpara[0]
        height = bgpara[1]
        numax = bgpara[2]
        sigma = bgpara[3]

        ampHarvey, freqHarvey, powerHarvery = [], [], []
        for igran in range(granNumber):
            ampHarvey.append(bgpara[4+igran*2])
            freqHarvey.append(bgpara[4+igran*2+1])
            powerHarvery.append(bgpara[4+igran*2+2])

        zeta = 2.0*2.0**0.5/np.pi

        power_gran = np.zeros(len(self.freq))
        for igran in range(granNumber):
            power_gran += zeta*ampHarvey[igran]**2.0/(freqHarvey[igran]*(1+(self.freq/freqHarvey[igran])**powerHarvery[igran]))
        power_gaussian = height * np.exp(-1.0*(numax-self.freq)**2/(2.0*sigma**2.0))

        if bgtype == "withgaussian":
            power = power_gran + power_gaussian
        elif bgtype == "withoutgaussian":
            power = power_gran

        power *= self.__response_function()
        power += flatNoiseLevel
        return power

    def __response_function(self):
        sincfunctionarg = (np.pi/2.0)*self.freq/self.fnyq
        responsefunction = (np.sin(sincfunctionarg)/sincfunctionarg)**2.0
        return responsefunction

    def __guess_background_parameters(self, granNumber=1):
        zeta = 2*2**0.5/np.pi
        flatNoiseLevel = np.median(self.powers[int(len(self.freq)*0.9):])
        height = se.closest_point(self.freq, self.powers, self.numax) 
        sigma = 3.0 * self.dnu

        freqHarvey_solar = [2440.5672465, 735.4653975, 24.298031575000003]
        numax_solar = 3050
        ampHarvey, freqHarvey= [], []
        for igran in range(granNumber):
            freqHarvey.append(self.numax/numax_solar*freqHarvey_solar[igran])
            ampHarvey.append((se.closest_point(self.freq, self.powers, freqHarvey[igran])*2/zeta*freqHarvey[igran])**0.5)

        init = [flatNoiseLevel, height, self.numax, sigma]
        bounds = [[flatNoiseLevel*0.9, flatNoiseLevel*1.1], 
                  [height*0.2, height*5.0],
                  [self.numax*0.8, self.numax*1.2],
                  [sigma*0.5, sigma*4.0]]
        names = ["W", "H", "numax", "sigma"]
        for igran in range(granNumber):
            init.append(ampHarvey[igran])
            init.append(freqHarvey[igran])
            init.append(4.0)
            bounds.append([ampHarvey[igran]*0.3, ampHarvey[igran]*3.0])
            bounds.append([freqHarvey[igran]*0.2, freqHarvey[igran]*5.0])
            bounds.append([2.0, 8.0])
            names.append("a"+str(igran))
            names.append("b"+str(igran))
            names.append("c"+str(igran))
        return init, bounds, names

    def fit_background(self, granNumber=1, ifdisplay=True):

        assert granNumber in [1,2,3], "granNumber should be one 1, 2 or 3."
        assert ("freq" in self.__dict__) & ("power" in self.__dict__), "Power spectrum must be passed in before any fitting."


        def residuals_bg(bgpara):
            model = self.__background_model(bgpara, bgtype="withgaussian", granNumber=granNumber)
            return np.sum(np.log(model) + self.power/model)
        self.__smooth_power()
        init, bounds, names = self.__guess_background_parameters(granNumber=granNumber)
        res = minimize(residuals_bg, init, bounds=bounds)
        bgpara = res.x
        
        # save backbround parameters
        print("Background parameters "+", ".join(names))
        print(bgpara)
        np.savetxt(self.outputdir+"bgpara.txt", bgpara, 
                   header=", ".join(names))       

        power_bg = self.__background_model(bgpara, bgtype="withoutgaussian", granNumber=granNumber)
        power_bg_wg = self.__background_model(bgpara, bgtype="withgaussian", granNumber=granNumber)

        # divide the background and save power spectrum
        self.snr = self.power/power_bg
        SNRData = np.array([self.freq, self.snr]).T
        np.save(self.outputdir+"snr", SNRData)

        # plot: power spectrum,  1 - log, 2 - linear
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(211)
        ax1.plot(self.freq, self.power, color='gray')
        ax1.plot(self.freq, self.powers, color='black')
        ax1.plot(self.freq, power_bg_wg, color='green', linestyle='--')
        ax1.plot(self.freq, power_bg, color='green')
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.axis([np.min(self.freq), np.max(self.freq), np.min(self.powers), np.max(self.power)])

        ax2 = fig.add_subplot(212)
        ax2.plot(self.freq, self.snr, color='gray')
        ax2.plot(self.freq, self.powers/power_bg, color='black')
        index = np.all([self.freq > self.numax-4.*self.dnu,
                        self.freq < self.numax+4.*self.dnu], axis=0)

        _xmin = max(np.min(self.freq), self.numax-9.*self.dnu)
        _xmax = min(np.max(self.freq), self.numax+9.*self.dnu)
        ax2.axis([_xmin, _xmax, 0., np.max((self.powers/power_bg)[index])])
        ax2.axhline(1.0, color="green")

        plt.savefig(self.outputdir+"snr.png")
        #plt.show()
        plt.close()

        return
    
#     def check_background(self):
#         return


   
    
    
    


