import numpy as np
from Models import *

class FitParameters:
	def __init__(self, mode_freq, mode_l, freq, power, powers, dnu,
				ifSplit=False, inclination=None, 
				ifResolved=True, resolution=None,
				ifVaryLwPerMode=True, ifVaryFsPerMode=True, ifVaryAmpPerMode=True):

		self.mode_l = mode_l
		self.mode_freq = mode_freq
		self.freq = freq
		self.power = power
		self.powers = powers
		self.dnu = dnu
		self.ifSplit = ifSplit
		self.inclination = inclination
		self.ifResolved = ifResolved
		self.resolution = resolution
		self.ifVaryLwPerMode=ifVaryLwPerMode
		self.ifVaryFsPerMode=ifVaryFsPerMode
		self.ifVaryAmpPerMode=ifVaryAmpPerMode

		n_mode = mode_l.shape[0]
		ifIncludeModes = (n_mode != 0)
		ifFreeInclination = (ifSplit & (inclination is None))
		ifSetUniformFs = (not ifVaryFsPerMode) & ((mode_l[mode_l>=1].shape[0]>=1) & ifSplit)
		ifSetUniformLw = (not ifVaryLwPerMode)
		ifSetUniformAmp = (not ifVaryAmpPerMode)
		if ifIncludeModes:
			paraNamesInBlock = [{} for i in range(n_mode)]

			for imode in range(n_mode):
				simode=str(imode)
				if ifResolved:
					paras = {}
					if ifVaryAmpPerMode:
						paras["amp"] = "amp"+simode
					if ifVaryLwPerMode:
						paras["lw"] = "lw"+simode
					if (ifVaryFsPerMode & (mode_l[imode] >=1) & ifSplit):
						paras["fs"] = "fs"+simode
					paras["fc"] = "fc"+simode
					paraNamesInBlock[imode] = paras
				else:
					if ifVaryAmpPerMode:
						paras["height"] = "height"+simode
					paras["fc"] = "fc"+simode
					paraNamesInBlock[imode] = paras

			names = {}
			if ifSetUniformAmp:
				names["amp"] = "amp"
			if ifSetUniformLw:
				names["lw"] = "lw"
			if ifSetUniformFs:
				names["fs"] = "fs"
			if ifFreeInclination:
				names["i"] = "i"
			paraNamesInBlock.append(names)

			self.paraNamesInBlock = paraNamesInBlock
		else:
			self.paraNamesInBlock = [{"bg":"bg"}]


		self.n_mode = n_mode
		self.n_mode_l0 = np.where(mode_l==0)[0].shape[0]

		self.ifIncludeModes = ifIncludeModes
		self.ifFreeInclination = ifFreeInclination

		self.paraNames = np.concatenate([list(block.values()) for block in self.paraNamesInBlock])
		self.nParas = self.paraNames.shape[0]
		
		return

class Priors(FitParameters):
	def __init__(self, FitParametersObj,
				priorGuess=None, initGuess=None,
				ampPrior="flat_prior",
				lwPrior="flat_prior",
				fsPrior="flat_plus_gaussian_decaying_wing_prior",
				fcPrior="flat_prior",
				iPrior="flat_prior",
				heightPrior="flat_prior",
				bgPrior="flat_prior"):
		"""
		Under devolopement:
		1. add support for "guassian_prior"
		2. add support for "lorentzian_prior"
		3. develop a smart way to determine the range of a mode
		"""

		FitParameters.__init__(self, FitParametersObj.mode_freq, 
								FitParametersObj.mode_l, 
								FitParametersObj.freq,
								FitParametersObj.power, 
								FitParametersObj.powers,
								FitParametersObj.dnu,
								ifSplit=FitParametersObj.ifSplit, 
								inclination=FitParametersObj.inclination, 
								ifResolved=FitParametersObj.ifResolved,
								resolution=FitParametersObj.resolution,
								ifVaryLwPerMode=FitParametersObj.ifVaryLwPerMode,
								ifVaryFsPerMode=FitParametersObj.ifVaryFsPerMode,
								ifVaryAmpPerMode=FitParametersObj.ifVaryAmpPerMode)
		
		self.ampPrior=ampPrior
		self.lwPrior=lwPrior
		self.fsPrior=fsPrior
		self.fcPrior=fcPrior
		self.iPrior=iPrior
		self.heightPrior=heightPrior
		self.bgPrior=bgPrior
		self.priorGuess = priorGuess
		self.initGuess = initGuess
		if (priorGuess is None) | (initGuess is None):
			pg, ig = self._guess_prior()
			self.priorGuess = pg if priorGuess is None else priorGuess
			self.initGuess = ig if initGuess is None else initGuess

		self.prior_guess = [item  for blocks in self.priorGuess  for item in blocks.values()]
		self.init_guess = [list(blocks.values()) for blocks in self.initGuess]

		return


	def _guess_prior(self):
		if self.ifIncludeModes:
			priorGuess = [[] for i in range(self.n_mode)]
			initGuess = [[] for i in range(self.n_mode)]
			for imode in range(self.n_mode):
				keys = self.paraNamesInBlock[imode].keys()
				tmode_freq = self.mode_freq[imode]
				self._guess_prior_initialize(mode_freq=tmode_freq)
				prior, init = {}, {}
				for key in keys:
					if key == "amp":
						pr, ini = self._guess_prior_amp(self.ampPrior)
					elif key == "lw":
						pr, ini = self._guess_prior_lw(self.lwPrior)
					elif key == "fs":
						pr, ini = self._guess_prior_fs(self.fsPrior)
					elif key == "fc":
						pr, ini = self._guess_prior_fc(self.fcPrior)
					elif key == "height":
						pr, ini = self._guess_prior_height(self.heightPrior)
					prior[key] = pr
					init[key] = ini
				priorGuess[imode] = prior
				initGuess[imode] = init

			keys = self.paraNamesInBlock[-1].keys()
			tmode_freq = self.mode_freq[0]
			# tmode_l = self.mode_l[0]
			self._guess_prior_initialize(mode_freq=tmode_freq)
			prior, init = {}, {}
			for key in keys:
				if key == "amp":
					pr, ini = self._guess_prior_amp(self.ampPrior)
				elif key == "lw":
					pr, ini = self._guess_prior_lw(self.lwPrior)
				elif key == "fs":
					pr, ini = self._guess_prior_fs(self.fsPrior)
				elif key == "fc":
					pr, ini = self._guess_prior_fc(self.fcPrior)
				elif key == "height":
					pr, ini = self._guess_prior_height(self.heightPrior)
				elif key == "i":
					pr, ini = self._guess_prior_i(self.iPrior)
				prior[key] = pr
				init[key] = ini	
			priorGuess.append(prior)
			initGuess.append(init)	

		else: 
			self._guess_prior_initialize()
			pr, ini = self._guess_prior_bg(self.bgPrior)
			priorGuess = [{"bg":pr}]
			initGuess = [{"bg":ini}]

		# print(priorGuess)
		# print(initGuess)

		return priorGuess, initGuess

	def _guess_prior_initialize(self, mode_freq=None):
		if not (mode_freq is None):
			factor = 0.04
			lmode_freq = self.mode_freq[self.mode_freq < mode_freq]
			lmode_freq = lmode_freq.max() if lmode_freq.shape[0] != 0 else mode_freq-factor*self.dnu

			umode_freq = self.mode_freq[self.mode_freq > mode_freq]
			umode_freq = umode_freq.min() if umode_freq.shape[0] != 0 else mode_freq+factor*self.dnu		

			lowerbound = max(lmode_freq, mode_freq-factor*self.dnu, np.min(self.freq))
			upperbound = min(umode_freq, mode_freq+factor*self.dnu, np.max(self.freq))

			idx = (self.freq >= lowerbound) & (self.freq <= upperbound)
			tfreq, tpowers = self.freq[idx], self.powers[idx]

			height = np.max(tpowers)-1.0
			height = height if height>0 else np.max(tpowers)
			dfreq = np.median(tfreq[1:]-tfreq[:-1])
			area = np.sum(tpowers*dfreq-1.0*dfreq)
			lw = 2.0*area/height/np.pi if area>0 else 1.0
			amp = (height*np.pi*lw)**0.5

			self._height, self._amp, self._fc, self._lw = height, amp, mode_freq, lw
			# self._bg_min, self._bg_max = np.min(tpowers), np.max(tpowers)
			self._lowerbound, self._upperbound = lowerbound, upperbound

		else:
			self._bg_min, self._bg_max = np.min(self.powers), np.max(self.powers)
			lowerbound, upperbound = np.min(self.freq), np.max(self.freq)
			self._lowerbound, self._upperbound = lowerbound, upperbound

		return lowerbound, upperbound

	def _guess_prior_amp(self, ampPrior):
		if ampPrior == "flat_prior":
			return [self._amp*0.2, self._amp*5.0], self._amp
		elif ampPrior == "flat_plus_gaussian_decaying_wing_prior":
			return [self._amp*0.2, self._amp*5.0, self._amp], self._amp

	def _guess_prior_lw(self, lwPrior):
		if lwPrior == "flat_prior":
			return [self._lw*0.2, self._lw*5.0], self._lw
		elif lwPrior == "flat_plus_gaussian_decaying_wing_prior":
			return [self._lw*0.2, self._lw*5.0, self._lw], self._lw

	def _guess_prior_fc(self, fcPrior):
		if fcPrior == "flat_prior":
			return [self._lowerbound, self._upperbound], self._fc
		elif fcPrior == "flat_plus_gaussian_decaying_wing_prior":
			return [self._lowerbound, self._upperbound, 1.0], self._fc

	def _guess_prior_fs(self, fsPrior):
		if fsPrior == "flat_prior":
			return [0., .5], 0.1
		elif fsPrior == "flat_plus_gaussian_decaying_wing_prior":
			return [0., .5, 0.5], 0.1

	def _guess_prior_i(self, iPrior):
		if iPrior == "flat_prior":
			return [-np.pi/2.0, np.pi], np.pi/2.0
		elif iPrior == "flat_plus_gaussian_decaying_wing_prior":
			assert False, "inclination is not compatible with prior ``flat_plus_gaussian_decaying_wing_prior''. "

	def _guess_prior_height(self, heightPrior):
		if heightPrior == "flat_prior":
			return [self._height*0.2, self._height*5.0], self._height
		elif heightPrior == "flat_plus_gaussian_decaying_wing_prior":
			return [self._height*0.2, self._height*5.0, self._height], self._height

	def _guess_prior_bg(self, bgPrior):
		if bgPrior == "flat_prior":
			return [self._bg_min, self._bg_max], 1.
		elif bgPrior == "flat_plus_gaussian_decaying_wing_prior":
			return [self._bg_min, self._bg_max, 1.], 1.

	def flat_prior(self, theta, lowerLimit, upperLimit):
		if not lowerLimit <= theta <= upperLimit:
			return -np.inf
		else:
			return np.log(1.0/(upperLimit - lowerLimit))

	def flat_plus_gaussian_decaying_wing_prior(self, theta, flatStart, flatEnd, gaussianSigma):
		if theta < flatStart:
			return -np.inf
		else:
			height = 1.0/(flatEnd + (2*np.pi)**0.5 * gaussianSigma/2.0 - flatStart)
			if theta < flatEnd:
				lnprior = np.log(height)
			else:
				lnprior = np.log(height) - (theta-flatEnd)**2.0/(2*gaussianSigma**2.0)
			return lnprior

	def lnprior(self, theta):
		lnprior = 0.
		itheta = 0

		for iblock, block in enumerate(self.priorGuess):
			for key in block:
				params = (theta[itheta],) + tuple(block[key])
				if key == "amp":
					lnprior += getattr(self, self.ampPrior)(*params)
				elif key == "lw":
					lnprior += getattr(self, self.lwPrior)(*params)
				elif key == "fs":
					lnprior += getattr(self, self.fsPrior)(*params)
				elif key == "fc":
					lnprior += getattr(self, self.fcPrior)(*params)
				elif key == "height":
					lnprior += getattr(self, self.heightPrior)(*params)
				elif key == "i":
					lnprior += getattr(self, self.iPrior)(*params)
				elif key == "bg":
					lnprior += getattr(self, self.bgPrior)(*params)
				itheta += 1
			
		return lnprior




class Likelihoods(FitParameters):
	def __init__(self, FitParametersObj,
				fnyq):

		FitParameters.__init__(self, FitParametersObj.mode_freq, 
								FitParametersObj.mode_l, 
								FitParametersObj.freq,
								FitParametersObj.power, 
								FitParametersObj.powers,
								FitParametersObj.dnu,
								ifSplit=FitParametersObj.ifSplit, 
								inclination=FitParametersObj.inclination, 
								ifResolved=FitParametersObj.ifResolved,
								resolution=FitParametersObj.resolution,
								ifVaryLwPerMode=FitParametersObj.ifVaryLwPerMode,
								ifVaryFsPerMode=FitParametersObj.ifVaryFsPerMode,
								ifVaryAmpPerMode=FitParametersObj.ifVaryAmpPerMode)
		
		self.fnyq=fnyq

		return

	def model(self, theta, x=None):
		if x is None: x = self.freq
		model = np.zeros(x.shape[0])
		itheta = 0

		if self.ifIncludeModes:
			commonParaNames = (self.paraNamesInBlock[-1]).keys()
			nname = len(commonParaNames)
			commonParaValues = {}
			for iname, name in enumerate(commonParaNames):
				commonParaValues[name] = theta[-nname+iname]

			for imode in range(self.n_mode):
				privateParaNames = (self.paraNamesInBlock[imode]).keys()
				nname = len(privateParaNames)
				privateParaValues = {}
				for iname, name in enumerate(privateParaNames):
					privateParaValues[name] = theta[itheta]
					itheta += 1
				paraValues = {**commonParaValues, **privateParaValues}
				if self.ifResolved:
					if (self.mode_l[imode] >=1) & self.ifSplit:
						# ["amp", "lw", "fs", "fc"]

						if self.ifFreeInclination:
							inclination = paraValues["i"]
						else:
							inclination = self.inclination

						modelParameters=[paraValues["amp"], paraValues["lw"],
										paraValues["fs"], paraValues["fc"], 
										inclination]
						
						model += lorentzian_splitting_model(x, modelParameters, self.fnyq, self.mode_l[imode])

					else:
						# ["amp", "lw", "fc"]
						modelParameters=[paraValues["amp"], paraValues["lw"],
										0., paraValues["fc"], 
										0.]
						model += lorentzian_splitting_model(x, modelParameters, self.fnyq, 0)
				else:
					# ["height", "fc"]
					modelParameters=[paraValues["height"], paraValues["fc"]]
					model += sinc_model(x, modelParameters, self.fnyq, self.resolution)
			model += 1.
		else:
			# [["bg"]]
			modelParameters = [theta[itheta]]
			model += flat_model(x, modelParameters, self.fnyq)
		return model

	def lnlikelihood(self, theta):
		model = self.model(theta)
		return -np.sum(np.log(model) + self.power/model)

	def minus_lnlikelihood(self, theta):
		model = self.model(theta)
		return np.sum(np.log(model) + self.power/model)


class Posteriors(Priors, Likelihoods):
	def __init__(self, FitParametersObj, PriorsObj, LikelihoodsObj):
		Priors.__init__(self, FitParametersObj,
				priorGuess=PriorsObj.priorGuess, initGuess=PriorsObj.initGuess,
				ampPrior=PriorsObj.ampPrior,
				lwPrior=PriorsObj.lwPrior,
				fsPrior=PriorsObj.fsPrior,
				fcPrior=PriorsObj.fcPrior,
				iPrior=PriorsObj.iPrior,
				heightPrior=PriorsObj.heightPrior,
				bgPrior=PriorsObj.bgPrior)
		Likelihoods.__init__(self, FitParametersObj, LikelihoodsObj.fnyq)
		return

	def lnpost(self, theta):
		lnprior = self.lnprior(theta)
		if not np.isfinite(lnprior):
			return -np.inf
		else:
			lnlikelihood = self.lnlikelihood(theta)
			return lnprior + lnlikelihood


