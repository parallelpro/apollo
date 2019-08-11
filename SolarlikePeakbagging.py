import numpy as np
import matplotlib.pyplot as plt
import os

# customized
from scipy.optimize import minimize, basinhopping
from scipy.signal import find_peaks


from Bayesian import FitParameters, Priors, Likelihoods, Posteriors
from Fitter import FitModes, PTSampler, ESSampler, LSSampler

class SolarlikePeakbagging:
	"""docstring for SolarlikePeakbagging"""
	def __init__(self, starname, outputdir, fnyq, numax):
		"""
		Docstring
    	"""

		# super(SolarlikePeakbagging, self).__init__()
		self._sep = "\\" if os.name=="nt" else "/"
		self._starname = starname
		self._outputdir = outputdir  # "with a / in the end"
		assert outputdir.endswith(self._sep), "outputdir should end with "+self._sep
		self._fnyq = fnyq  # in microHz (muHz) 

		# numax and dnu are only approximations
		self._numax0 = numax  # in microHz (muHz)
		self._dnu0 = (self._numax0/3050)**0.77 * 135.1 # Stello+2009 

		# nu_max and delta_nu are accruate values

		return


	def parse_power_spectrum(self, freq, power, trimUpperLimitInDnu=None, 
    	 trimLowerLimitInDnu=None, ifSmooth=False):
		"""
		Pass the power spectrum in.

		Input:

		freq: np.array
			frequency in muHz.

		power: np.array
			the background divided power spectrum (so now is s/b instead).


		Optional input:


		"""

		assert len(freq) == len(power), "len(freq) != len(power)"

		idx = np.array(np.zeros(len(freq))+1, dtype=bool)
		freq = np.array(freq)
		power = np.array(power)
		if not trimUpperLimitInDnu is None:
			idx = (idx) & (freq<=self._numax0+trimUpperLimitInDnu*self._dnu0)
		if not trimLowerLimitInDnu is None:
			idx = (idx) & (freq>=self._numax0-trimLowerLimitInDnu*self._dnu0)

		self.freq = freq[idx]
		self.power = power[idx]
		ifSmooth: self.powers = self._smooth_power()
		return


	def _trim_power_spectrum(self, freq, power, powers=None, trimUpperLimitInDnu=None,
		 trimLowerLimitInDnu=None):
		"""
		Trim the power spectrum.

		Input:

		freq: np.array
			frequency in muHz.

		power: np.array
			the background divided power spectrum (so now is s/b instead).


		Optional input:


		"""
		idx = np.array(np.zeros(len(freq))+1, dtype=bool)
		freq = np.array(freq)
		power = np.array(power)
		if not trimUpperLimitInDnu is None:
			idx = (idx) & (freq<=self._numax0+trimUpperLimitInDnu*self._dnu0)
		if not trimLowerLimitInDnu is None:
			idx = (idx) & (freq>=self._numax0-trimLowerLimitInDnu*self._dnu0)

		if powers is None:
			return freq[idx], power[idx]
		else:
			return freq[idx], power[idx], powers[idx]


	def _smooth_power(self, period=None):
		if period is None: period = self._dnu0/15.0 # microHz
		self.powers = self._smooth_wrapper(self.freq, self.power, period, "bartlett")
		return

	def _smooth_wrapper(self, x, y, period, windowtype, samplinginterval=None):

		if samplinginterval is None: samplinginterval = np.median(x[1:-1] - x[0:-2])
		if not windowtype in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
			raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

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


	def guess_ppara(self, fixDnu=None):
		"""
		Docstring
		"""

		# smooth the power spectrum
		self._smooth_power()
        
		# slice the power spectrum
		freq, power, powers = self._trim_power_spectrum(self.freq, self.power, 
			powers=self.powers, trimUpperLimitInDnu=3., trimLowerLimitInDnu=3.)


		def ppara_model(ppara):
			# initialize
			x = freq
			ymodel = np.zeros(len(freq))
			if fixDnu is None: 
				dnu02, dnu, eps = ppara
			else:
				dnu02, eps = ppara
				dnu = fixDnu
			_xmin, _xmax = np.min(x), np.max(x)
			# print(_xmin, _xmax, eps)
			n_p = np.arange(int(_xmin/dnu-eps-1), int(_xmax/dnu-eps+1), 1)
			nu_l0 = dnu*(n_p+eps)
			nu_l2 = dnu*(n_p+eps)-dnu02
			nu_l1 = dnu*(n_p+eps)+0.5*dnu

			# l=0 template
			for inu in nu_l0:
				lw, center, maxima = 0.04*dnu, inu, 1.0
				idx = (x>inu-lw) & (x<inu+lw)
				ymodel[idx] = -(1.0/lw**2.0)*(x[idx] - center)**2.0 + maxima

			# l=2 template
			for inu in nu_l2:
				lw, center, maxima = 0.03*dnu, inu, 0.6
				idx = (x>inu-lw) & (x<inu+lw)
				ymodel[idx] = -(1.0/lw**2.0)*(x[idx] - center)**2.0 + maxima

			# l=1 template
			for inu in nu_l1:
				lw, center, maxima = 0.03*dnu, inu, 1.0
				idx = (x>inu-lw) & (x<inu+lw)
				ymodel[idx] = -(1.0/lw**2.0)*(x[idx] - center)**2.0 + maxima

			ymodel[ymodel<0] = 0.

			return ymodel

		def corr_ppara(ppara):
			ymodel = ppara_model(ppara)
			y = (powers-1)/np.max(powers-1)
			return -np.log(np.sum(ymodel*y)/np.sum(ymodel))*10.0

		# set free parameters
		if fixDnu is None:
			init = [self._dnu0/10., self._dnu0, 0.5]
			bounds = [[self._dnu0/20., self._dnu0/8.],
					  [self._dnu0*0.8, self._dnu0*1.2],
					  [0.0001, 1.-0.0001]]
			names = ["dnu02", "dnu", "eps"]
		else:
			init = [self._dnu0/10., 0.5]
			bounds = [[self._dnu0/20., self._dnu0/8.],
					  [0.0001, 1.-0.0001]]
			names = ["dnu02", "eps"]

		minimizer_kwargs = {"bounds":bounds}
		res = basinhopping(corr_ppara, init, minimizer_kwargs=minimizer_kwargs)
		ppara = res.x			

		if fixDnu is None:
			self._dnu02, self.dnu, self.eps = ppara
		else:
			self._dnu02, self.eps = ppara
			self.dnu = fixDnu
		self.print_ppara()
		self.print_ppara_tofile()

		# plot - power spectrum
		fig = plt.figure(figsize=(8,5))
		ax1 = fig.add_subplot(111)
		ymodel = ppara_model(ppara)
		yobs = (powers-1)/np.max(powers-1)
		ax1.plot(freq, ymodel, color="green")
		ax1.plot(freq, yobs, color="black")
		plt.savefig(self._outputdir+"ppara.png")

		return

    
	def set_ppara_fromfile(self, inputfile=None):
		"""
		Docstring
		"""

		if inputfile is None: inputfile = self._outputdir + "ppara.txt"

		self._dnu02, self.dnu, self.eps = np.loadtxt(inputfile, delimiter=",")
		return


	def set_ppara(self, dnu02, dnu, eps):
		"""
		Docstring
		"""
		self._dnu02, self.dnu, self.eps = dnu02, dnu, eps
		return
    

	def print_ppara(self):
		"""
		Docstring
		"""
		print("dnu02 = ", self._dnu02)
		print("dnu = ", self.dnu)
		print("eps = ", self.eps)
		return


	def print_ppara_tofile(self):
		"""
		Docstring
		"""
		outputfile = self._outputdir + "ppara.txt"
		print("Writing ppara to "+outputfile)
		np.savetxt(outputfile, np.array([[self._dnu02, self.dnu, self.eps]]),
		           header="dnu02, dnu, eps", delimiter=",")
		return


	def guess_modeid(self, trimLowerLimitInDnu=9.0, trimUpperLimitInDnu=9.0,
		 height=2.0, prominence=1.5):
		""" 
		An initial guess for all mode frequencies in the power spectrum.
		After running this function, you should visually check the power
		spectrum and see if the identified modes generated from the code
		are correct (matched with your expectations).		


		Input:


		Optional input:

		trimLowerLimitInDnu: float, default: 9.0
			the lower boundary of the power spectrum slice, in unit of dnu.

		trimUpperLimitInDnu: float, default: 9.0
			the upper boundary of the power spectrum slice, in unit of dnu.

		height: float, default: 2.0
			the minimum height for a peak to be recognised, in unit of power.

		prominence: float, default: 1.5
			the minimum prominence for a peak to be recognised, in unit of power.


		Output:

		Files containing necessary outputs.
		1. table frequencyGuess.csv

		Under development:
		1. Improvement to mode identification - slide with spectrum and define probs.

		"""

		# smooth the power spectrum
		self._smooth_power(period=1.)
        
		# slice the power spectrum
		freq, power, powers = self._trim_power_spectrum(self.freq, self.power, 
			powers=self.powers, trimUpperLimitInDnu=trimLowerLimitInDnu, 
			trimLowerLimitInDnu=trimUpperLimitInDnu)

		dnu02, dnu, eps, numax = self._dnu02, self.dnu, self.eps, self._numax0
		samplinginterval = np.median(freq[1:]-freq[:-1])

		# assign l=0,1,2 region to the power spectrum
		rfreq = freq/dnu % 1.0
		lowc = [-dnu02/dnu/2.0, +0.10, -dnu02/dnu-0.05]
		highc = [+0.10, 1.0-dnu02/dnu-0.05, -dnu02/dnu/2.0]
		idx_l = []
		for l in range(3):
			dum1 = (rfreq>=eps+lowc[l]) & (rfreq<eps+highc[l])
			dum2 = (rfreq>=eps+lowc[l]-1) & (rfreq<eps+highc[l]-1)
			dum3 = (rfreq>=eps+lowc[l]+1) & (rfreq<eps+highc[l]+1)
			idx_l.append((dum1|dum2|dum3))


		# slice power spectrum into blocks
		n_blocks = int(trimLowerLimitInDnu+trimUpperLimitInDnu)+1
		# label_echx, label_echy, label_text = [[] for i in range(3)]
		rfreq_init = (numax/dnu)%1.0
		if rfreq_init-eps < 0.0: freq_init = numax-dnu*trimLowerLimitInDnu-dnu+np.abs(rfreq_init-eps)*dnu-dnu02-0.05*dnu
		if rfreq_init-eps >=0.0: freq_init = numax-dnu*trimLowerLimitInDnu-np.abs(rfreq_init-eps)*dnu-dnu02-0.05*dnu

		mode_l, mode_freq = [], []
		# find peaks in each dnu range
		for iblock in range(n_blocks):
			freq_low, freq_high = freq_init+iblock*dnu, freq_init+(iblock+1)*dnu
			idx_norder = np.all(np.array([freq>=freq_low,freq<freq_high]),axis=0)
			
			# find peaks in each l range
			tidx_l, tmode_freq, tmode_l  = [], [], []
			for l in range(3):
				tidx_l.append(np.all(np.array([freq>=freq_low,freq<freq_high,idx_l[l]]),axis=0))
				if len(freq[tidx_l[l]])==0: continue
				tfreq, tpower, tpowers = freq[tidx_l[l]], power[tidx_l[l]], powers[tidx_l[l]]
				meanlevel = np.median(tpowers)
				# find the highest peak in this range as a guess for the radial mode
				idx_peaks, properties = find_peaks(tpowers, height=(height,None), 
					distance=int(dnu02/samplinginterval/5.0), prominence=(prominence,None))
				Npeaks = len(idx_peaks)
				if Npeaks != 0:
					if l != 1:
						idx_maxpeak = idx_peaks[properties["peak_heights"] == properties["peak_heights"].max()]
						tmode_freq.append(tfreq[idx_maxpeak[0]])
						tmode_l.append(l)
					else:
						for ipeak in range(Npeaks):
							tmode_freq.append(tfreq[idx_peaks[ipeak]])
							tmode_l.append(l)
			tmode_freq, tmode_l = np.array(tmode_freq), np.array(tmode_l)
			mode_freq.append(tmode_freq)
			mode_l.append(tmode_l)

		# save a table
		# but first let's associate each mode with a group number
		mode_freq_group, mode_l_group, mode_group = [], [], np.array([])
		idx = np.argsort(mode_freq)
		mode_freq, mode_l = mode_freq[idx], mode_l[idx]
		dist = mode_freq[1:] - mode_freq[:-1]
		group_idx = np.where(dist>=0.2*dnu)[0] + 1 #each element the new group start from 
		Ngroups = len(group_idx) + 1
		group_idx = np.insert(group_idx,0,0)
		group_idx = np.append(group_idx,len(mode_freq))

		# just sort a bit
		for igroup in range(Ngroups):
			tmode_freq = mode_freq[group_idx[igroup]:group_idx[igroup+1]]
			tmode_l = mode_l[group_idx[igroup]:group_idx[igroup+1]]

			mode_freq_group.append(tmode_freq)
			mode_l_group.append(tmode_l)
			elements = group_idx[igroup+1] - group_idx[igroup]
			for j in range(elements):
				mode_group = np.append(mode_group,igroup)

		mode_group = np.array(mode_group, dtype=int)
		mode_freq = np.concatenate(mode_freq_group)
		mode_l = np.concatenate(mode_l_group)

		idx = np.lexsort((mode_freq,mode_l))
		mode_group, mode_freq, mode_l = mode_group[idx], mode_freq[idx], mode_l[idx]

		table = np.array([np.arange(len(mode_freq)), np.zeros(len(mode_freq))+1, 
			mode_group, mode_l, mode_freq]).T
		np.savetxt(self._outputdir+"frequencyGuess.csv", table, delimiter=",", fmt=("%d","%d","%d","%d","%10.4f"), 
			header="mode_id, ifpeakbagging, igroup, mode_l, mode_freq")

		return

	def set_modeid_fromfile(self, inputfile=None):
		"""
		Docstring
		"""

		if inputfile is None: inputfile=self._outputdir+"frequencyGuess.csv"

		dtype = [("mode_id", "int"),
				("ifpeakbagging", "int"),
				("igroup", "int"), 
				("mode_l", "int"),
				("mode_freq", "float")]
		arraylist = np.genfromtxt(inputfile, 
			delimiter=",", skip_header=1, dtype=dtype)
		self.modeInputTable = arraylist[arraylist["ifpeakbagging"]==1]
		return


	def plot_modeid(self, trimLowerLimitInDnu=9.0, trimUpperLimitInDnu=9.0,):
		""" 
		Plot the initial guess for all mode frequencies in the power spectrum.
		After running this function, you should visually check the power
		spectrum and see if the identified modes generated from the code
		are correct (matched with your expectations).		


		Input:


		Optional input:

		trimLowerLimitInDnu: float, default: 9.0
			the lower boundary of the power spectrum slice, in unit of dnu.

		trimUpperLimitInDnu: float, default: 9.0
			the upper boundary of the power spectrum slice, in unit of dnu.


		Output:

		Files containing necessary outputs.
		1. analysis plot frequencyGuess.png

		"""

		# set up
		dnu02, dnu, eps, numax = self._dnu02, self.dnu, self.eps, self._numax0
		if eps < 0.5:
			offset = eps*dnu-dnu02-0.05*dnu
		elif eps >= 0.5:
			offset = -eps*dnu-dnu02-0.05*dnu
		mode_l, mode_freq = self.modeInputTable["mode_l"], self.modeInputTable["mode_freq"]

		# set up plot
		fig = plt.figure(figsize=(15,12))

		# smooth the power spectrum
		self._smooth_power(period=1.)
        
		# slice the power spectrum
		freq, power, powers = self._trim_power_spectrum(self.freq, self.power, 
			powers=self.powers, trimUpperLimitInDnu=9., 
			trimLowerLimitInDnu=9.)

		# assign l=0,1,2 region to the power spectrum
		rfreq = freq/dnu % 1.0
		lowc = [-dnu02/dnu/2.0, +0.10, -dnu02/dnu-0.05]
		highc = [+0.10, 1.0-dnu02/dnu-0.05, -dnu02/dnu/2.0]
		idx_l = []
		for l in range(3):
			dum1 = (rfreq>=eps+lowc[l]) & (rfreq<eps+highc[l])
			dum2 = (rfreq>=eps+lowc[l]-1) & (rfreq<eps+highc[l]-1)
			dum3 = (rfreq>=eps+lowc[l]+1) & (rfreq<eps+highc[l]+1)
			idx_l.append((dum1|dum2|dum3))		

		# slice power spectrum into blocks
		n_blocks = int(trimLowerLimitInDnu+trimUpperLimitInDnu)+1
		label_echx, label_echy, label_text = [[] for i in range(3)]
		rfreq_init = (numax/dnu)%1.0
		if rfreq_init-eps < 0.0: freq_init = numax-dnu*trimLowerLimitInDnu-dnu+np.abs(rfreq_init-eps)*dnu-dnu02-0.05*dnu
		if rfreq_init-eps >=0.0: freq_init = numax-dnu*trimLowerLimitInDnu-np.abs(rfreq_init-eps)*dnu-dnu02-0.05*dnu


		for iblock in range(n_blocks):
			freq_low, freq_high = freq_init+iblock*dnu, freq_init+(iblock+1)*dnu
			idx_norder = np.all(np.array([freq>=freq_low,freq<freq_high]),axis=0)

			# labels on the right side of the echelle
			label_text.append("{:0.0f}".format(iblock))
			label_echx.append(2.01*dnu)
			# py = (freq_high-0.1*dnu) - ((freq_high-offset-0.1*dnu) % dnu) - dnu/2.0
			py = freq_high-dnu/2.0-dnu
			label_echy.append(py)

			if len(np.where(idx_norder == True)[0])==0:
				continue

			tidx_l = []
			for l in range(3):
				tidx_l.append(np.all(np.array([freq>=freq_low,freq<freq_high,idx_l[l]]),axis=0))
			idx = (mode_freq > freq_low) & (mode_freq <= freq_high)
			tmode_freq, tmode_l = mode_freq[idx], mode_l[idx]

			### visulization (right)
			# ax1: the whole dnu range
			ax1 = fig.add_subplot(n_blocks,2,2*n_blocks-2*iblock)
			ax1.plot(freq[idx_norder], power[idx_norder], color="gray")
			ax1.plot(freq[tidx_l[0]], powers[tidx_l[0]], color="C0", linewidth=1)
			ax1.plot(freq[tidx_l[1]], powers[tidx_l[1]], color="C3", linewidth=1)
			ax1.plot(freq[tidx_l[2]], powers[tidx_l[2]], color="C2", linewidth=1)
			ax1.text(1.1, 0.5, str(iblock), ha="center", va="center", transform=ax1.transAxes, 
				bbox=dict(facecolor='white', edgecolor="black"))

			# label the mode candidates
			colors=["C0","C3","C2","C1"]
			markers=["o", "^", "s", "v"]
			c, d = ax1.get_ylim()
			Npeaks, Npeaks1 = len(tmode_freq), len(tmode_freq[tmode_l==1])
			for ipeak in range(Npeaks):
				ax1.scatter([tmode_freq[ipeak]],[c+(d-c)*0.8], c=colors[tmode_l[ipeak]], 
					marker=markers[tmode_l[ipeak]], zorder=10)
			### end of visulization


		### visulization (left) - plot echelle and collapsed echelle to locate peak
		# ax1 = plt.subplot2grid((5,3), (0,0), rowspan=4)
		ax1 = fig.add_subplot(1,2,1)
		echx, echy, echz = se.echelle(freq, powers, dnu, freq.min(), freq.max(), 
			echelletype="single", offset=offset)
		levels = np.linspace(np.min(echz), np.max(echz), 500)
		ax1.contourf(echx, echy, echz, cmap="gray_r", levels=levels)
		ax1.axis([np.min(echx), np.max(echx), np.min(echy), np.max(echy)])
		ax1.axvline(dnu, color="C0")
		# labels on the right side of the echelle
		for iblock in range(n_blocks):
			ax1.text(label_echx[iblock], label_echy[iblock], label_text[iblock],
			 verticalalignment="center")
		ax1.set_ylabel("Frequency [muHz]")


		# mark mode candidates on the echelle
		px = (mode_freq-offset) % dnu
		py = (mode_freq-offset) - ((mode_freq-offset) % dnu) + dnu/2.0 #+ offset
		for l in range(4):
			if len(px[mode_l==l]) == 0: continue
			ax1.plot(px[mode_l==l], py[mode_l==l], "x", color=colors[l])
			ax1.plot(px[mode_l==l]+dnu, py[mode_l==l]-dnu, "x", color=colors[l])

		# save plot
		plt.savefig(self._outputdir+"frequencyGuess.png")
		plt.close()

		return

	def fit_mode(self,
		igroup=None, ifTestH1=False,
		ifResolved=True, resolution=None,
		ifSplit=False, inclination=None,
		ifVaryLwPerMode=True, ifVaryFsPerMode=True, ifVaryAmpPerMode=True,
		trimLowerLimitInDnu=None,
		trimUpperLimitInDnu=None,
		fitType="LeastSquare", nsteps=None, ifOutputSamples=False,
		priorsKwargs={}, likelihoodsKwargs={}, samplerKwargs={}):

		"""
		Provide a wrapper to fit modes defined in mode_freq.

		Input:

		mode_freq: np.ndarray(N,)
			the mode frequencies intend to fit, in muHz.

		mode_l: np.ndarray(N,)
			the mode degree corresponding to mode_freq.
			now only support 0, 1, 2, and 3.

		inclination: float
			the inclination angle, in rad.


		Optional input:

		fitType: str, default: "ParallelTempering"
			one of ["ParallelTempering", "Ensemble", "LeastSquare"].

		ifOutputSamples: bool, default: False
			set True to output MCMC sampling points.

		trimLowerLimitInDnu: float, default: None
			trim the data into [min(mode_freq)-trimLowerLimitInDnu*dnu,
			max(mode_freq)+trimUpperLimitInDnu*dnu] for fit.

		trimUpperLimitInDnu: float, default: None
			trim the data into [min(mode_freq)-trimLowerLimitInDnu*dnu,
			max(mode_freq)+trimUpperLimitInDnu*dnu] for fit.

		nsteps: int, default: 2000
			the number of steps to iterate for mcmc run.

		ifResolved: bool, default: True
			whether the modes are resolved. pass a 1-d array (len(mode_freq),)
			containing True/False.

		resolution: float, default: None
			the frequency spectra resolution. must be set when passing values
			from ``ifResolved''.


		Output:

		Data: acceptance fraction, bayesian evidence, 
			parameter estimation result, parameter initial guess.
		Plots: fitting results, posterior distribution, traces.

		"""

		# specifiy which group of modes to fit
		if igroup is None: 
			groups = np.unique(self.modeInputTable["igroup"])
		else:
			if type(igroup) is int: 
				groups = np.array([igroup], dtype=int)
			else:
				groups = np.array(igroup, dtype=int)


		# fit
		for igroup in groups: #igroup
			# for imode in modes: #mode_freq, mode_l
			table = self.modeInputTable[self.modeInputTable["igroup"]==igroup]
			mode_freq, mode_l = table["mode_freq"], table["mode_l"]
			# idx = np.lexsort((mode_freq, mode_l))
			# mode_freq, mode_l = mode_freq[idx], mode_l[idx]

			# split in subgroups
			cdata, data = self._fit_prep(mode_freq, mode_l, 	
							igroup=igroup, ifTestH1=ifTestH1,
							ifResolved=ifResolved, resolution=resolution,
							ifSplit=ifSplit, inclination=inclination,
							ifVaryLwPerMode=ifVaryLwPerMode, ifVaryFsPerMode=ifVaryFsPerMode, ifVaryAmpPerMode=ifVaryAmpPerMode,
							trimLowerLimitInDnu=trimLowerLimitInDnu,
							trimUpperLimitInDnu=trimUpperLimitInDnu,
							fitType=fitType)

			# fit in different subgroups to test H1 hypothesis (mode significance)
			for tdata in data:	
				self._fit(cdata, tdata,
				fitType=fitType, nsteps=nsteps, ifOutputSamples=ifOutputSamples,
				priorsKwargs=priorsKwargs, likelihoodsKwargs=likelihoodsKwargs,
				samplerKwargs=samplerKwargs)

		return


	def _fit_prep(self, mode_freq, mode_l,
		igroup=None, ifTestH1=False,
		ifResolved=True, resolution=None,
		ifSplit=False, inclination=None,
		ifVaryLwPerMode=True, ifVaryFsPerMode=True, ifVaryAmpPerMode=True,
		trimLowerLimitInDnu=None,
		trimUpperLimitInDnu=None,
		fitType="LeastSquare"):
		'''
		under development:
		1. change the way to specify trimLowerLimitInDnu - derive this parameter from data

		'''


		# check
		assert mode_freq.shape[0] == mode_l.shape[0], "mode_freq and mode_l does not match in dimension."
		assert fitType in ["ParallelTempering", "Ensemble", "LeastSquare"], "fitType should be one of ['ParallelTempering', 'Ensemble', 'LeastSquare']"


		class datacube:
			def __init__(self):
				pass

		cdata = datacube()

		# essentials
		dnu, fnyq = self.dnu, self._fnyq
		freq, power, powers = self.freq, self.power, self.powers
		cdata.fnyq = fnyq

		# specify output directory
		filepath = self._outputdir+self._sep+"pkbg"+self._sep
		if not os.path.exists(filepath): os.mkdir(filepath)
		filepath = filepath+str(igroup)+self._sep
		if not os.path.exists(filepath): os.mkdir(filepath)


		# specify if modes are resolved
		if  (ifResolved is False):
			assert not (resolution is None), "resolution is not set."
		# else:
		# 	if np.where(ifResolved==False)[0].shape[0] != 0 :
		# 		assert not (resolution is None), "resolution is not set."

		cdata.ifResolved = ifResolved
		cdata.resolution = resolution


		# specify if the modes are splitted, and if so, if inclination is set free.
		ifFreeInclination = (inclination is None) & ifSplit
		if not ifSplit: inclination = 0.

		cdata.ifSplit = ifSplit
		cdata.ifFreeInclination = ifFreeInclination
		cdata.inclination = inclination


		cdata.ifVaryLwPerMode = ifVaryLwPerMode
		cdata.ifVaryFsPerMode = ifVaryFsPerMode
		cdata.ifVaryAmpPerMode = ifVaryAmpPerMode

		# specify the range of the power spectra used to fit
		# a very radical case - more suitable for red giants
		if trimLowerLimitInDnu is None: 
			minl=mode_l[mode_freq == mode_freq.min()][0]
			if minl==0: trimLowerLimitInDnu=0.08
			if minl==1: trimLowerLimitInDnu=0.05
			if minl==2: trimLowerLimitInDnu=0.20
			if minl>=3: trimLowerLimitInDnu=0.05
		if trimUpperLimitInDnu is None:
			maxl=mode_l[mode_freq == mode_freq.max()][0]
			if maxl==0: trimUpperLimitInDnu=0.20
			if maxl==1: trimUpperLimitInDnu=0.05
			if maxl==2: trimUpperLimitInDnu=0.08
			if maxl>=3: trimUpperLimitInDnu=0.05

		trimLowerLimitInDnu *= dnu
		trimUpperLimitInDnu *= dnu

		# trim data into range we use
		# this is for plot
		idx = (freq >= np.min(mode_freq)-0.5*dnu) & (freq <= np.max(mode_freq)+0.5*dnu)
		freq, power, powers = freq[idx], power[idx], powers[idx]
		cdata.freq = freq
		cdata.power = power
		cdata.powers = powers

		# this is for fit
		idx = (freq >= np.min(mode_freq)-trimLowerLimitInDnu) & (freq <= np.max(mode_freq)+trimUpperLimitInDnu)
		tfreq, tpower, tpowers = freq[idx], power[idx], powers[idx]
		cdata.tfreq = tfreq
		cdata.tpower = tpower
		cdata.tpowers = tpowers


		# initilize
		n_mode = mode_l.shape[0]
		n_mode_l0 = np.where(mode_l == 0)[0].shape[0]

		# specify if test H1
		if ifTestH1: assert fitType == "ParallelTempering", "to test H1 hypothesis, fitype must be set to PT."
		n_subgroups = mode_freq.shape[0] if ifTestH1 else 0

		data = []

		# subgroup = 0, the fit which includes all modes
		tdata = datacube()

		tfilepath = filepath+"0"+self._sep
		if not os.path.exists(tfilepath): os.mkdir(tfilepath)
		tdata.filepath = tfilepath
		tdata.mode_freq = mode_freq
		tdata.mode_l = mode_l
		data.append(tdata)


		# isubgroup, the fit which one of the mode is missing
		for isubgroup in range(1,n_subgroups+1):
			tdata = datacube()

			tfilepath = filepath+str(isubgroup)+self._sep
			if not os.path.exists(tfilepath): os.mkdir(tfilepath)
			tdata.filepath = tfilepath
			idx = np.ones(mode_freq.shape[0], dtype=bool)
			idx[isubgroup-1] = False
			tdata.mode_freq = mode_freq[idx]
			tdata.mode_l = mode_l[idx]
			data.append(tdata)

		return cdata, data



	def _fit(self, cdata, data,  
		fitType="ParallelTempering", nsteps=None, ifOutputSamples=False,
		priorsKwargs={}, likelihoodsKwargs={}, samplerKwargs={}):

		"""
		Under development:
		0. specify if the modes are splitted, and if so, if inclination is set free. ok
		1. add support for ifTestH1 -> add or remove a mode. ok
		2. specify if a mode is resolved -> change lorentz model to sinc. ok
		3. ability to customize likelihood and prior. partially ok

		"""

		fnyq = cdata.fnyq

		ifSplit = cdata.ifSplit
		inclination = cdata.inclination
		ifResolved =  cdata.ifResolved
		resolution = cdata.resolution

		# used for plot
		freq = cdata.freq
		power = cdata.power
		powers =  cdata.powers

		# used for fit
		tfreq = cdata.tfreq
		tpower = cdata.tpower
		tpowers = cdata.tpowers

		filepath = data.filepath
		mode_freq = data.mode_freq
		mode_l = data.mode_l

		ifFreeInclination = cdata.ifFreeInclination
		ifVaryLwPerMode = cdata.ifVaryLwPerMode
		ifVaryFsPerMode = cdata.ifVaryFsPerMode
		ifVaryAmpPerMode = cdata.ifVaryAmpPerMode

		dnu = self.dnu

		fitParameters = FitParameters(mode_freq, mode_l, tfreq, tpower, tpowers, dnu,
				ifSplit=ifSplit, inclination=inclination, 
				ifResolved=ifResolved, resolution=resolution,
				ifVaryLwPerMode=ifVaryLwPerMode,
				ifVaryFsPerMode=ifVaryFsPerMode,
				ifVaryAmpPerMode=ifVaryAmpPerMode)

		if fitType=="LeastSquare": priorsKwargs={"ampPrior":"flat_prior",
				"lwPrior":"flat_prior","fsPrior":"flat_prior",
				"fcPrior":"flat_prior","iPrior":"flat_prior",
				"heightPrior":"flat_prior","bgPrior":"flat_prior"}
		priors = Priors(fitParameters, **priorsKwargs)
		likelihoods = Likelihoods(fitParameters, fnyq, **likelihoodsKwargs)
		posteriors = Posteriors(fitParameters, priors, likelihoods)

		fitModes = FitModes(fitParameters, priors, likelihoods, posteriors,
				freq, power, powers)

		if fitType=="ParallelTempering":
			sampler = PTSampler(fitModes, filepath, **samplerKwargs)
		elif fitType=="Ensemble":
			sampler = ESSampler(fitModes, filepath, **samplerKwargs)
		elif fitType=="LeastSquare":
			sampler = LSSampler(fitModes, filepath, **samplerKwargs)

		sampler.run()
		sampler.output()

		return

	def summarize_peakbagging():
		"""
		Docstring
		"""
		pass

