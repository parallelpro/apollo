#!/usr/bin/env/ python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


# customized
from scipy.optimize import minimize, basinhopping
import emcee
import corner
import sys 

from .toolkit import return_2dmap_axes, plot_mcmc_traces

# under development / improvement:
# 1. use smart ways to output data
# 2. add support for fixing or constraining relative fcs/amps/lws/fss/...

__all__ = ["FitModes", "PTSampler", "ESSampler", "LSSampler"]

class FitModes:
	def __init__(self, FitParametersObj, PriorsObj, LikelihoodsObj, PosteriorsObj,
				freq_for_plot, power_for_plot, powers_for_plot):
		self.FitParametersObj = FitParametersObj
		self.PriorsObj = PriorsObj
		self.LikelihoodsObj = LikelihoodsObj
		self.PosteriorsObj = PosteriorsObj
		self.freq = freq_for_plot
		self.power = power_for_plot
		self.powers = powers_for_plot
		return

	def _plot_fit_results(self, freq, power, powers, tfreq, power_guess, power_fit,
							mode_freq, mode_l, dnu, priorGuess):

		numberOfSquareBlocks = mode_freq.shape[0] if mode_freq.shape[0] != 0 else 1
		fig, axes = self._return_2dmap_axes(numberOfSquareBlocks)
		color = ["blue", "red", "green", "purple"]
		marker = ["o", "^", "s", "v"]

		for i in range(numberOfSquareBlocks):
			ax = axes[i]
			ax.plot(freq, power, color="lightgray", label="power")
			ax.plot(freq, powers, color="black", label="smooth")
			ax.plot(freq, power_guess, color="blue", label="guess")
			ax.plot(freq, power_fit, color="orange", label="fit")
			# ax.legend()

			if mode_freq.shape[0] != 0:
				a, b = mode_freq[i] - 0.2*dnu, mode_freq[i] + 0.2*dnu
				idx = (freq > a) & (freq < b)
				c, d = np.min(power[idx]), np.max(power[idx])

				ax.scatter([mode_freq[i]],[c+(d-c)*0.8], c=color[mode_l[i]], marker=marker[mode_l[i]])
				ax.errorbar([mode_freq[i]],[c+(d-c)*0.8], ecolor=color[mode_l[i]],
					 xerr=[[np.abs(priorGuess[i]["fc"][0]-mode_freq[i])],
					 [np.abs(priorGuess[i]["fc"][1]-mode_freq[i])]], capsize=5)

				ax.axis([a, b, c, d])
				ax.axvline(np.min(tfreq), linestyle="--", color="gray")
				ax.axvline(np.max(tfreq), linestyle="--", color="gray")		

		for ax in axes[i+1:]:
			fig.delaxes(ax)

		return fig

	def _return_2dmap_axes(self, numberOfSquareBlocks):
		return return_2dmap_axes(numberOfSquareBlocks)

	def _plot_mcmc_traces(self, ndim, samples, para_names):
		return plot_mcmc_traces(ndim, samples, para_names)


class PTSampler(FitModes):
	def __init__(self, FitModesObj, filepath, nsteps=2000,
					nburn=1000, ntemps=20, nwalkers=100,
					ifOutputSamples=False):
		FitModes.__init__(self, FitModesObj.FitParametersObj, 
							FitModesObj.PriorsObj, 
							FitModesObj.LikelihoodsObj, 
							FitModesObj.PosteriorsObj,
							FitModesObj.freq, 
							FitModesObj.power, 
							FitModesObj.powers)

		self.nburn=nburn
		self.nsteps=nsteps
		self.ntemps=ntemps
		self.nwalkers=nwalkers
		self.filepath=filepath
		self.para_guess = np.concatenate(self.PriorsObj.init_guess)
		self.ndim=self.FitParametersObj.nParas
		self.ifOutputSamples=ifOutputSamples

		return

	def _display_bar(self, j, nburn, width=30):
		n = int((width+1) * float(j) / nburn)
		sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
		return

	def run(self):
		print("enabling ParallelTempering sampler.")
		pos0 = [[self.para_guess + 1.0e-8*np.random.randn(self.ndim) for j in range(self.nwalkers)] for k in range(self.ntemps)]
		sampler = emcee.PTSampler(self.ntemps, self.nwalkers, self.ndim, self.LikelihoodsObj.lnlikelihood, self.PriorsObj.lnprior)

		# burn-in
		print("start burning in. nburn:", self.nburn)
		for j, result in enumerate(sampler.sample(pos0, iterations=self.nburn, thin=10)):
			self._display_bar(j, self.nburn)
		sys.stdout.write("\n")
		pos, lnpost, lnlike = result
		sampler.reset()

		# actual iteration
		print("start iterating. nsteps:", self.nsteps)
		for j, result in enumerate(sampler.sample(pos, iterations=self.nsteps)):#, lnprob0=lnpost, lnlike0=lnlike
			self._display_bar(j, self.nsteps)
		sys.stdout.write("\n")

		# modify samples
		samples = sampler.chain[0,:,:,:].reshape((-1,self.ndim))

		# fold the parameter space of i
		if self.FitParametersObj.ifFreeInclination: 
			idx = samples[:,-1]<0.
			samples[idx,-1]=-samples[idx,-1]
			idx = samples[:,-1]>np.pi/2.
			samples[idx,-1]=np.pi-samples[idx,-1]
		self.samples = samples

		# save evidence
		self.evidence = sampler.thermodynamic_integration_log_evidence() 
		print("Bayesian evidence lnZ: {:0.5f}".format(self.evidence[0]))
		print("Bayesian evidence error dlnZ: {:0.5f}".format(self.evidence[1]))

		# save estimation result
		# 16, 50, 84 quantiles
		result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
			zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
		self.para_fit = result[:,0]

		# maximum
		para_fitmax = np.zeros(self.ndim)
		for ipara in range(self.ndim):
			n, bins, _ = plt.hist(samples[:,ipara], bins=80)
			idx = np.where(n == n.max())[0][0]
			para_fitmax[ipara] = bins[idx:idx+1].mean()
		self.para_fitmax = para_fitmax

		self.result = np.concatenate([result, para_fitmax.reshape(self.ndim,1)], axis=1)

		# save acceptance fraction
		self.acceptance_fraction = np.array([np.mean(sampler.acceptance_fraction)])
		print("Mean acceptance fraction: {:0.3f}".format(self.acceptance_fraction[0]))

		return

	def output(self):
		st = "PT"
		# save evidence
		np.savetxt(self.filepath+st+"evidence.txt", self.evidence, delimiter=",", fmt=("%0.8f"), header="bayesian_evidence")

		# save samples if the switch is toggled on
		if self.ifOutputSamples: np.save(self.filepath+st+"samples.npy", self.samples)

		# save guessed parameters
		np.savetxt(self.filepath+st+"guess.txt", self.para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

		# plot triangle and save
		para_names = self.FitParametersObj.paraNames
		fig = corner.corner(self.samples, labels=para_names, quantiles=(0.16, 0.5, 0.84), truths=self.para_fitmax)
		fig.savefig(self.filepath+st+"triangle.png")
		plt.close()

		# save estimation result
		np.savetxt(self.filepath+st+"summary.txt", self.result, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f", "%0.8f"), header="50th quantile, 16th quantile sigma, 84th quantile sigma, maximum")

		# save mean acceptance rate
		np.savetxt(self.filepath+st+"acceptance_fraction.txt", self.acceptance_fraction, delimiter=",", fmt=("%0.8f"), header="acceptance_fraction")

		# plot traces and save
		fig = self._plot_mcmc_traces(self.ndim, self.samples, para_names)
		plt.savefig(self.filepath+st+'traces.png')
		plt.close()

		# # plot fitting results and save
		# power_fit = self.LikelihoodsObj.model(self.para_fit, x=self.freq)
		# power_guess = self.LikelihoodsObj.model(self.para_guess, x=self.freq)
		# fig = self._plot_fit_results(self.freq, self.power, self.powers, self.FitParametersObj.freq, power_guess, power_fit,
		# 					self.FitParametersObj.mode_freq, self.FitParametersObj.mode_l, self.FitParametersObj.dnu, 
		# 					self.PriorsObj.priorGuess)
		# plt.savefig(self.filepath+st+"fitmedian.png")
		# plt.close()

		# power_fitmax = self.LikelihoodsObj.model(self.para_fitmax, x=self.freq)
		# fig = self._plot_fit_results(self.freq, self.power, self.powers, self.FitParametersObj.freq, power_guess, power_fitmax,
		# 					self.FitParametersObj.mode_freq, self.FitParametersObj.mode_l, self.FitParametersObj.dnu, 
		# 					self.PriorsObj.priorGuess)
		# plt.savefig(self.filepath+st+"fitmax.png")
		# plt.close()

		return


class ESSampler(FitModes):
	def __init__(self, FitModesObj, filepath, nsteps=2000,
					nburn=1000, nwalkers=100,
					ifOutputSamples=False):
		FitModes.__init__(self, FitModesObj.FitParametersObj, 
							FitModesObj.PriorsObj, 
							FitModesObj.LikelihoodsObj, 
							FitModesObj.PosteriorsObj,
							FitModesObj.freq, 
							FitModesObj.power, 
							FitModesObj.powers)

		self.nburn=nburn
		self.nsteps=nsteps
		self.nwalkers=nwalkers
		self.filepath=filepath
		self.para_guess = np.concatenate(self.PriorsObj.init_guess)
		self.ndim=self.FitParametersObj.nParas
		self.ifOutputSamples=ifOutputSamples

		return

	def _display_bar(self, j, nburn, width=30):
		n = int((width+1) * float(j) / nburn)
		sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
		return

	def run(self):
		# run mcmc with ensemble sampler
		print("enabling Ensemble sampler.")
		pos0 = [self.para_guess + 1.0e-8*np.random.randn(self.ndim) for j in range(self.nwalkers)]
		sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.PosteriorsObj.lnpost)

		# burn-in
		print("start burning in. nburn:", self.nburn)
		for j, result in enumerate(sampler.sample(pos0, iterations=self.nburn, thin=10)):
			self._display_bar(j, self.nburn)
		sys.stdout.write("\n")
		pos, lnpost, rstate = result
		sampler.reset()

		# actual iteration
		print("start iterating. nsteps:", self.nsteps)
		for j, result in enumerate(sampler.sample(pos, iterations=self.nsteps)):#, lnprob0=lnpost
			self._display_bar(j, self.nsteps)
		sys.stdout.write("\n")

		# modify samples
		samples = sampler.chain[:,:,:].reshape((-1,self.ndim))

		# fold the parameter space of i
		if self.FitParametersObj.ifFreeInclination: 
			idx = samples[:,-1]<0.
			samples[idx,-1]=-samples[idx,-1]
			idx = samples[:,-1]>np.pi/2.
			samples[idx,-1]=np.pi-samples[idx,-1]
		self.samples = samples

		# save estimation result
		# 16, 50, 84 quantiles
		result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
			zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
		self.para_fit = result[:,0]

		# maximum
		para_fitmax = np.zeros(self.ndim)
		for ipara in range(self.ndim):
			n, bins, _ = plt.hist(samples[:,ipara], bins=80)
			idx = np.where(n == n.max())[0][0]
			para_fitmax[ipara] = bins[idx:idx+1].mean()
		self.para_fitmax = para_fitmax

		self.result = np.concatenate([result, para_fitmax.reshape(self.ndim,1)], axis=1)

		# save acceptance fraction
		self.acceptance_fraction = np.array([np.mean(sampler.acceptance_fraction)])
		print("Mean acceptance fraction: {:0.3f}".format(self.acceptance_fraction[0]))

		return

	def output(self):
		st = "ES"

		# save samples if the switch is toggled on
		if self.ifOutputSamples: np.save(self.filepath+st+"samples.npy", self.samples)

		# save guessed parameters
		np.savetxt(self.filepath+st+"guess.txt", self.para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

		# plot triangle and save
		para_names = self.FitParametersObj.paraNames
		fig = corner.corner(self.samples, labels=para_names, quantiles=(0.16, 0.5, 0.84), truths=self.para_fitmax)
		fig.savefig(self.filepath+st+"triangle.png")
		plt.close()

		# save estimation result
		np.savetxt(self.filepath+st+"summary.txt", self.result, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f", "%0.8f"), header="50th quantile, 16th quantile sigma, 84th quantile sigma, maximum")

		# save mean acceptance rate
		np.savetxt(self.filepath+st+"acceptance_fraction.txt", self.acceptance_fraction, delimiter=",", fmt=("%0.8f"), header="acceptance_fraction")

		# plot traces and save
		fig = self._plot_mcmc_traces(self.ndim, self.samples, para_names)
		plt.savefig(self.filepath+st+'traces.png')
		plt.close()

		# plot fitting results and save
		power_fit = self.LikelihoodsObj.model(self.para_fit, x=self.freq)
		power_guess = self.LikelihoodsObj.model(self.para_guess, x=self.freq)
		fig = self._plot_fit_results(self.freq, self.power, self.powers, self.FitParametersObj.freq, power_guess, power_fit,
							self.FitParametersObj.mode_freq, self.FitParametersObj.mode_l, self.FitParametersObj.dnu, 
							self.PriorsObj.priorGuess)
		plt.savefig(self.filepath+st+"fitmedian.png")
		plt.close()

		power_fitmax = self.LikelihoodsObj.model(self.para_fitmax, x=self.freq)
		fig = self._plot_fit_results(self.freq, self.power, self.powers, self.FitParametersObj.freq, power_guess, power_fitmax,
							self.FitParametersObj.mode_freq, self.FitParametersObj.mode_l, self.FitParametersObj.dnu, 
							self.PriorsObj.priorGuess)
		plt.savefig(self.filepath+st+"fitmax.png")
		plt.close()

		return


class LSSampler(FitModes):
	def __init__(self, FitModesObj, filepath):
		FitModes.__init__(self, FitModesObj.FitParametersObj, 
							FitModesObj.PriorsObj, 
							FitModesObj.LikelihoodsObj, 
							FitModesObj.PosteriorsObj,
							FitModesObj.freq, 
							FitModesObj.power, 
							FitModesObj.powers)

		self.filepath=filepath
		self.para_guess = np.concatenate(self.PriorsObj.init_guess)
		self.ndim = self.FitParametersObj.nParas

		return

	def run(self):
		# maximize likelihood function by scipy.optimize.minimize function
		bounds = (np.concatenate([self.PriorsObj.prior_guess])).tolist()
		# print(bounds)
		# print(self.para_guess)
		minimizer_kwargs={"bounds":bounds}
		result = basinhopping(self.LikelihoodsObj.minus_lnlikelihood, self.para_guess, minimizer_kwargs=minimizer_kwargs)
		para_fit = result.x
		if self.FitParametersObj.ifFreeInclination: 
			if para_fit[-1]<0.: para_fit[-1]=-para_fit[-1]
			if para_fit[-1]>np.pi/2.: para_fit[-1]=np.pi-para_fit[-1]
		self.para_fit = para_fit

		return

	def output(self):
		st = "LS"
		# save guessed parameters
		np.savetxt(self.filepath+st+"guess.txt", self.para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

		# save estimation result
		np.savetxt(self.filepath+st+"summary.txt", self.para_fit, delimiter=",", fmt=("%0.8f"), header="parameter")

		# plot fitting results and save
		power_fit = self.LikelihoodsObj.model(self.para_fit, x=self.freq)
		power_guess = self.LikelihoodsObj.model(self.para_guess, x=self.freq)
		fig = self._plot_fit_results(self.freq, self.power, self.powers, self.FitParametersObj.freq, power_guess, power_fit,
							self.FitParametersObj.mode_freq, self.FitParametersObj.mode_l, self.FitParametersObj.dnu, 
							self.PriorsObj.priorGuess)
		plt.savefig(self.filepath+st+"fit.png")
		plt.close()

		return


class MySampler(FitModes):
	def __init__(self, FitModesObj, filepath):
		"""
		Fitmodes.__init__(self, FitModesObj.FitParametersObj, 
							FitModesObj.PriorsObj, 
							FitModesObj.LikelihoodsObj, 
							FitModesObj.PosteriorsObj,
							FitModesObj.freq, 
							FitModesObj.power, 
							FitModesObj.powers)

		self.filepath=filepath
		self.para_guess = np.concatenate(self.PriorsObj.init_guess)
		self.ndim=self.FitParametersObj.nParas

		return
		"""
		pass

	def run(self):
		"""
		sampler = SomeSampler(self.LikelihoodsObj.lnlikelihood, self.PriorsObj.lnprior, self.PosteriorsObj.lnpost)
		sampler.run()
		self.para_fit = sampler.results

		return
		"""
		pass

	def output(self):
		"""
		# save guessed parameters
		np.savetxt(self.filepath+"guess.txt", self.para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

		# save estimation result
		np.savetxt(self.filepath+"summary.txt", self.para_fit, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f"), header="parameter, upper uncertainty, lower uncertainty")

		# plot fitting results and save
		power_fit = self.LikelihoodsObj.model(self.para_fit, x=self.freq)
		power_guess = self.LikelihoodsObj.model(self.para_guess, x=self.freq)
		fig = self._plot_fit_results(self.freq, self.power, self.powers, self.FitParametersObj.freq, power_guess, power_fit,
							self.FitParametersObj.mode_freq, self.FitParametersObj.mode_l, self.FitParametersObj.dnu, 
							self.FitParametersObj.n_mode, self.FitParametersObj.n_mode_l0,
							self.PriorsObj.prior_guess)
		plt.savefig(self.filepath+"fit.png")
		plt.close()

		return
		"""
		pass