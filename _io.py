import numpy as np

__all__ = ["fitGuess"]


class myStructure:

    def __init__(self, headers, data):
        # we do not indulge adults
        assert type(headers) is dict, "type of headers should be dict."
        assert type(data) is np.ndarray, 
                "type of data should be ndarray."
        assert (len(data.shape) is 1) & (not (data.dtype.names is None)), 
                "data should be structured array with accessible field names."
    
        self.headers = headers
        self.data = data
        self.nheaders = len(headers)
        self.ndata = len(ndata)
        return


    def write(self, file, listOfHeaderFormats, listOfDataFormats):
        """
        docstring
        """
        # write in self._outputdir+"frequencyGuess.txt"

        # with open(outputdir, "w+") as f:
        file.write("----------\n")
        file.write(", ".join(self.data.dtype.names)+"\n")
        np.savetxt(file, self.data, delimiter=",", fmt=", ".join(listOfDataFormats))

        return self


    def read(self, file):
        """
        docstring
        """
        # read in self._outputdir+"frequencyGuess.txt"

        # content = file.split("----------\n")

        dtype = [() for i in range(self.nheaders)]
                # [("mode_id", "int"),
                # ("ifpeakbagging", "int"),
                # ("igroup", "int"), 
                # ("mode_l", "int"),
                # ("mode_freq", "float")]
        arraylist = np.genfromtxt(inputfile, 
            delimiter=",", skip_header=1, dtype=dtype)

        for block in content:
            if block.startswith("#"): continue
            lines = block.split("\n")

            mode_group = int(lines[0].split(",")[-1])
            ifpeakbagging = int(lines[1].split(",")[-1])
            nmode = int(lines[2].split(",")[-1])

            arr = [line.split(",") for line in lines[3:3+nmode]]
            mode_l = np.array([a[0] for a in arr], dtype=int)
            mode_freq = np.array([a[1] for a in arr], dtype=float)
            mode_id = np.array([a[2] for a in arr], dtype=int)
            self.add_group(mode_group, mode_l, mode_freq, ifpeakbagging=ifpeakbagging, mode_id=mode_id)

        return self


if __name__ is "__main__":
    print("Hello.")



# class groupOfTable:
# 	def __init__(self, numberOfLabelsInGroup, numberOfParasInGroup, 
# 		numberOfLabelArraysInGroup, numberOfParaArraysInGroup):
# 	"""
# 	docstring
# 	"""
# 		self.numberOfLabelsInGroup = numberOfLabelsInGroup
# 		self.numberOfParasInGroup = numberOfParasInGroup
# 		self.numberOfLabelArraysInGroup = numberOfLabelArraysInGroup
# 		self.numberOfParaArraysInGroup = numberOfParaArraysInGroup
# 		return

# 	def add_group(self, listOfVarsInGroup, listOfArraysInGroup):







# class fitGuess:
# 	def __init__(self, outputdir):
# 		"""
# 		docstring
# 		"""

# 		self._outputdir = outputdir

# 		# integer
# 		self._mode_group = []
# 		self._ifpeakbagging = []
# 		self._nmode = []

# 		# np.array
# 		self._mode_l = []
# 		self._mode_freq = []
# 		self._mode_id = []

# 		return 


# 	def add_group(self, mode_group: int, mode_l: np.array, mode_freq: np.array, 
# 			ifpeakbagging: int=None, mode_id: np.array=None):
# 		"""
# 		docstring
# 		"""

# 		# automatic generate mode_id if None
# 		if mode_id is None:
# 			num = mode_l.shape[0]
# 			if len(self._mode_id) == 0:
# 				mode_id = np.arange(0, num, 1, dtype=int)
# 			else:
# 				start = np.concatenate(self._mode_id).max()+1
# 				mode_id = np.arange(start, start+num, 1, dtype=int)

# 		assert mode_l.shape == mode_freq.shape == mode_id.shape, "Shapes do not match."

# 		self._mode_group.append(int(mode_group))
# 		if ifpeakbagging is None: ifpeakbagging = 1
# 		self._ifpeakbagging.append(int(ifpeakbagging))
# 		self._nmode.append(mode_freq.shape[0])

# 		self._mode_l.append(np.array(mode_l, dtype=int))
# 		self._mode_freq.append(mode_freq)
# 		self._mode_id.append(np.array(mode_id, dtype=int))

# 		return self


# 	def write(self):
# 		"""
# 		docstring
# 		"""
# 		# write in self._outputdir+"frequencyGuess.txt"

# 		f = open(self._outputdir+"fitGuess.txt", "w+")
# 		f.write("# mode_l, mode_freq, mode_id\n")
# 		ngroups = len(self._mode_group)
# 		for igroup in range(ngroups):
# 			f.write("###\n")
# 			f.write("mode_group, {:0.0f}\n".format(self._mode_group[igroup]))
# 			f.write("ifpeakbagging, {:0.0f}\n".format(self._ifpeakbagging[igroup]))
# 			f.write("nmode, {:0.0f}\n".format(self._nmode[igroup]))
# 			for imode in range(self._nmode[igroup]):
# 				f.write("{:0.0f}, {:0.4f}, {:0.0f}\n".format(self._mode_l[igroup][imode], 
# 					self._mode_freq[igroup][imode],
# 					self._mode_id[igroup][imode]))
# 			f.write("\n")
# 		f.close()
# 		return self


# 	def read(self, inputfile=None):
# 		"""
# 		docstring
# 		"""
# 		# read in self._outputdir+"frequencyGuess.txt"

# 		if inputfile is None: inputfile = self._outputdir+"fitGuess.txt"

# 		with open(inputfile) as f:
# 			content = f.read().split("###\n")

# 		for block in content:
# 			if block.startswith("#"): continue
# 			lines = block.split("\n")

# 			mode_group = int(lines[0].split(",")[-1])
# 			ifpeakbagging = int(lines[1].split(",")[-1])
# 			nmode = int(lines[2].split(",")[-1])

# 			arr = [line.split(",") for line in lines[3:3+nmode]]
# 			mode_l = np.array([a[0] for a in arr], dtype=int)
# 			mode_freq = np.array([a[1] for a in arr], dtype=float)
# 			mode_id = np.array([a[2] for a in arr], dtype=int)
# 			self.add_group(mode_group, mode_l, mode_freq, ifpeakbagging=ifpeakbagging, mode_id=mode_id)

# 		return self


# 	def modify_group(self, mode_group: int, mode_l: np.array, mode_freq: np.array, 
# 			ifpeakbagging: int=None, mode_id: np.array=None):
# 		"""
# 		docstring
# 		"""
# 		idx = np.array(self._mode_group) != mode_group

# 		self._mode_group = self._mode_group[idx]
# 		self._ifpeakbagging = self._ifpeakbagging[idx]
# 		self._nmode = self._nmode[idx]
# 		self._mode_l = self._mode_l[idx]
# 		self._mode_freq = self._mode_freq[idx]
# 		self._mode_id = self._mode_id[idx]	

# 		self.add_group(mode_group, mode_l, mode_freq, 
# 			ifpeakbagging, mode_id)

# 		return self





# class fitGroupSummary:
# 	# initGuess, priorGuess, summary, configurations (ifSplit, inclination, ...)
# 	pass


# class fitSummary:
# 	pass


# class mcmcFitterGroupSummary:
# 	# acceptance fraction, evidence
# 	pass


# class FitterGroupLog:
# 	pass




# class Posteriors(Priors, Likelihoods):
# 	def __init__(self, FitParametersObj, PriorsObj, LikelihoodsObj):
# 		Priors.__init__(self, FitParametersObj,
# 				priorGuess=PriorsObj.priorGuess, initGuess=PriorsObj.initGuess,
# 				ampPrior=PriorsObj.ampPrior,
# 				lwPrior=PriorsObj.lwPrior,
# 				fsPrior=PriorsObj.fsPrior,
# 				fcPrior=PriorsObj.fcPrior,
# 				iPrior=PriorsObj.iPrior,
# 				heightPrior=PriorsObj.heightPrior,
# 				bgPrior=PriorsObj.bgPrior)
# 		Likelihoods.__init__(self, FitParametersObj, LikelihoodsObj.fnyq)
# 		return

# 	def lnpost(self, theta):
# 		lnprior = self.lnprior(theta)
# 		if not np.isfinite(lnprior):
# 			return -np.inf
# 		else:
# 			lnlikelihood = self.lnlikelihood(theta)
# 			return lnprior + lnlikelihood


