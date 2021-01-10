import numpy as np

__all__ = ["lorentzian_splitting_model", "sinc_model", "flat_model"]

def response_function(freq, fnyq):
	x = (np.pi/2.0)*freq/fnyq
	responsefunction = (np.sin(x)/x)**2.0
	return responsefunction

def lorentzian_splitting_model(freq, modelParameters, fnyq, mode_l):
	amplitude = modelParameters[0]
	linewidth = modelParameters[1]
	projectedSplittingFrequency = modelParameters[2]
	centralFrequency = modelParameters[3]
	inclination = modelParameters[4]
	splittingFrequency = projectedSplittingFrequency#/np.sin(inclination)
	height = amplitude**2.0/(np.pi*linewidth)

	responseFunction = response_function(freq, fnyq)

	power = np.zeros(len(freq))

	if mode_l == 0:
		power += height/(1.0 + (4.0*(freq-centralFrequency)**2.0/(linewidth**2.0)))
	if mode_l == 1:
		visibility_m0 = np.cos(inclination)**2.0
		visibility_m1 = np.sin(inclination)**2.0*0.5
		power += visibility_m0 * height/(1.0 + (4.0*(freq-centralFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency-splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency+splittingFrequency)**2.0/(linewidth**2.0)))
	if mode_l == 2:
		visibility_m0 = (3.0*np.cos(inclination)**2.0-1)**2.0*0.25
		visibility_m1 = np.sin(inclination*2.0)**2.0*3.0/8.0
		visibility_m2 = np.sin(inclination)**4.0*3.0/8.0
		power += visibility_m0 * height/(1.0 + (4.0*(freq-centralFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency-splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency+splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m2 * height/(1.0 + (4.0*(freq-centralFrequency-2.0*splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m2 * height/(1.0 + (4.0*(freq-centralFrequency+2.0*splittingFrequency)**2.0/(linewidth**2.0)))
	if mode_l == 3:
		visibility_m0 = pow(-3.0*np.cos(inclination)+5.0*pow(np.cos(inclination),3.0),2.0)/4.0
		visibility_m1 = 3.0/16.0*pow(np.sin(inclination),2.0)*pow(-1.0+5.0*pow(np.cos(inclination),2.0),2.0)
		visibility_m2 = 15.0/8.0*pow(np.cos(inclination),2.0)*pow(-1.0+pow(np.cos(inclination),2.0),2.0)
		visibility_m3 = 5.0/16.0*pow(1.0-pow(np.cos(inclination),2.0),3.0)
		power += visibility_m0 * height/(1.0 + (4.0*(freq-centralFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency-splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency+splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m2 * height/(1.0 + (4.0*(freq-centralFrequency-2.0*splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m2 * height/(1.0 + (4.0*(freq-centralFrequency+2.0*splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m3 * height/(1.0 + (4.0*(freq-centralFrequency-3.0*splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m3 * height/(1.0 + (4.0*(freq-centralFrequency+3.0*splittingFrequency)**2.0/(linewidth**2.0)))

	power *= responseFunction

	return power

def sinc_model(freq, modelParameters, fnyq, resolution):
	height = modelParameters[0]
	centralFrequency = modelParameters[1]

	responseFunction = response_function(freq, fnyq)

	unresolvedArgument = np.pi * (freq - centralFrequency) / resolution
	power = height * (np.sin(unresolvedArgument) / unresolvedArgument)**2.0
	power *= responseFunction

	return power

def flat_model(freq, modelParameters, fnyq):
	power = np.zeros(len(freq)) + modelParameters[0]
	responseFunction = response_function(freq, fnyq)
	power *= responseFunction
	return power