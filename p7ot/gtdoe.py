#
# coding: utf-8
#
# Copyright (C) DATADVANCE, 2016
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# along with this library.  If not, see <http://www.gnu.org/licenses/>.
#

import openturns as ot
from da.p7core import gtdoe
from .blackbox import _Blackbox


# Used to keep DoE settings in appropriate way (in terms of p7core.gtdoe)
class _DoeSettings(object):

    # {variable_name: option_name} dictionary
    # each option has it's name within constructors of techniques classes
    _SETTINGS_TO_OPTIONS = {
        # Basic options :
        'deterministic': 'GTDoE/Deterministic',
        'logLevel': 'GTDoE/LogLevel',
        'seed': 'GTDoE/Seed',
        # Advanced options :
        'categoricalVariables': 'GTDoE/CategoricalVariables',
        'maxParallel': 'GTDoE/MaxParallel',
        'technique': 'GTDoE/Technique',
        # Optimized Latin Hypercube Sampling options
        'iterations': 'GTDoE/OLHS/Iterations',
        # Sequential techniques options
        'leap': 'GTDoE/Sequential/Leap',
        'skip': 'GTDoE/Sequential/Skip',
        # Fractional Factorial options
        'generatingString': 'GTDoE/FractionalFactorial/GeneratingString',
        'mainFactors': 'GTDoE/FractionalFactorial/MainFactors',
        # Orthogonal Array options
        'levelsNumber': 'GTDoE/OrthogonalArray/LevelsNumber',
        'maxIterations': 'GTDoE/OrthogonalArray/MaxIterations',
        'multistartIterations': 'GTDoE/OrthogonalArray/MultistartIterations',
        # Adaptive DoE options
        'accelerator': 'GTDoE/Adaptive/Accelerator',
        'annealingCount': 'GTDoE/Adaptive/AnnealingCount',
        'criterion': 'GTDoE/Adaptive/Criterion',
        'exactFitRequired': 'GTDoE/Adaptive/ExactFitRequired',
        'initialCount': 'GTDoE/Adaptive/InitialCount',
        'initialDoeTechnique': 'GTDoE/Adaptive/InitialDoeTechnique',
        'internalValidation': 'GTDoE/Adaptive/InternalValidation',
        'oneStepCount': 'GTDoE/Adaptive/OneStepCount',
        'trainIterations': 'GTDoE/Adaptive/TrainIterations',
        # Box-Behnken design options
        'isFull': 'GTDOE/BoxBehnken/IsFull',
        # Optimal Design options
        'model': 'GTDoE/OptimalDesign/Model',
        'tries': 'GTDoE/OptimalDesign/Tries',
        'type': 'GTDoE/OptimalDesign/Type'
    }

    def __init__(self):
        self.__settings = {}
        self.__settings['options'] = {}
        self.__options = gtdoe.Generator().options

    def set_all(self, **kwargs):
        for key, value in kwargs.items():
            self.set(key=key, value=value)

    def set(self, key, value):
        option = self._SETTINGS_TO_OPTIONS.get(key)
        if option:
            self.__options.set(option, value)
        elif value:
            self.__settings[key] = value

    def get(self, key=None):
        if key is None:
            # return all settings (dictionary)
            self.__settings['options'] = self.__options.get()
            return self.__settings
        option = self._SETTINGS_TO_OPTIONS.get(key)
        if option:
            return self.__options.get(option)
        else:
            return self.__settings.get(key)


class _P7Experiment(ot.ExperimentImplementation):
    """
    Base class for da.p7core.gtdoe techniques

    Parameters
    ----------
    bounds: tuple(list[float], list[float])
        Design space bounds (lower, upper).
    count: int, long
        The number of points to generate
    technique: "RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq", "FullFactorial", "FractionalFactorial", "LHS", "OLHS",
               "OptimalDesign", "OrthogonalArray", "ParametricStudy", "BoxBehnken", "Adaptive" or "Auto"
        The generation algorithm to use (Default: "Auto").
    """

    def __init__(self, bounds, count, technique=None):
        self.__generator = gtdoe.Generator()
        self.__p7_result = None
        # preparing the parameters of p7core.gtdoe generate method
        self._settings = _DoeSettings()
        self.setBounds(bounds)
        self.setCount(count)
        self.setTechnique(technique)
        super(_P7Experiment, self).__init__()

    def generate(self):
        """
        Generate points according to the experiment settings.

        Returns
        -------
        sample : :class:`~openturns.NumericalSample`
            The points which constitute the design of experiments.
        """
        # use converted bounds for generate method
        self.__p7_result = self.__generator.generate(**self._settings.get())
        return ot.typ.NumericalSample(self.__p7_result.points)

    def getClassName(self):
        """
        Accessor to the object's name.

        Returns
        -------
        class_name : str
            The object class name (`object.__class__.__name__`).
        """
        return self.__class__.__name__

    def getBounds(self):
        """
        Get design space bounds

        Returns
        -------
        bounds : tuple(list[float], list[float])
            Design space bounds (lower, upper)
        """
        return self.__bounds

    def setBounds(self, bounds):
        """
        Set design space bounds

        Parameters
        -------
        bounds : tuple(list[float], list[float])
            Design space bounds (lower, upper)
        """
        self.__bounds = bounds
        # Convert bounds for p7 generator
        p7_bounds = self.__convert_bounds(bounds)
        self._settings.set('bounds', p7_bounds)

    def getCount(self):
        """
        Get the number of points to generate.

        Returns
        -------
        count: int, long
            The number of points to generate (not for the blackbox-based
            adaptive DoE; use budget).
        """
        return self._settings.get('count')

    def setCount(self, count):
        """
        Set the number of points to generate.

        Parameters
        ----------
        count: int, long
            The number of points to generate.
        """
        self._settings.set('count', count)

    def getTechnique(self):
        """
        Get the generation algorithm name.

        Returns
        -------
        technique: "RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq", "FullFactorial", "FractionalFactorial", "LHS",
                   "OLHS", "OptimalDesign", "OrthogonalArray", "ParametricStudy", "BoxBehnken", "Adaptive" or "Auto"
            The generation algorithm to use (Default: "Auto").
        """
        return self._settings.get('technique')

    def setTechnique(self, technique):
        """
        Set the generation algorithm.

        Parameters
        ----------
        technique: "RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq", "FullFactorial", "FractionalFactorial", "LHS",
                   "OLHS", "OptimalDesign", "OrthogonalArray", "ParametricStudy", "BoxBehnken", "Adaptive" or "Auto"
            The generation algorithm to use (Default: "Auto").
        """
        self._settings.set('technique', technique)

    def setLogger(self, logger):
        """
        Used to set up a logger for the DoE generation process.

        Parameters
        ----------
        logger : da.p7core.loggers.StreamLogger object
            See section Loggers in p7core documentation for details.
        """
        self.__generator.set_logger(logger)

    def setWatcher(self, watcher):
        """
        Used to set up a watcher for the DoE generation process.

        Parameters
        ----------
        watcher : da.p7core.watchers.DefaultWatcher object
            A watcher is an object that is capable of interrupting a process.
            See section Watchers in p7core documentation for details.
        """
        self.__generator.set_watcher(watcher)

    def getP7Result(self):
        """
        Accessor to p7core DoE result.

        Returns
        -------
        p7result : :class:`~da.p7core.gtdoe.Result`
            Result object.
        """
        return self.__p7_result

    def __convert_bounds(self, bounds):
        # Convert bounds from ot.Interval to (lowerBounds, upperBounds)
        return (list(bounds.getLowerBound()), list(bounds.getUpperBound()))


# Base class for DoE common options
class _Batch(_P7Experiment):
    def getDeterministic(self):
        """
        Get the value of Deterministic DoE setting

        Returns
        -------
        deterministic: boolean
            Require generation to be reproducible (Default: False).
        """
        return self._settings.get('deterministic')

    def setDeterministic(self, deterministic):
        """
        Set the value of Deterministic DoE setting

        Parameters
        ----------
        deterministic: boolean
            Require generation to be reproducible (Default: False).
        """
        self._settings.set('deterministic', deterministic)

    def getSeed(self):
        """
        Get the value of fixed random seed.

        Returns
        -------
        seed: integer in range [1, 2^31 - 1]
            Fixed random seed (Default:	100). This option sets fixed seed value,
            which is used in all randomized algorithms if deterministic option is on.
            If deterministic is off, the seed value is ignored.
        """

        return self._settings.get(seed)

    def setSeed(self, seed):
        """
        Set the value of fixed random seed.

        Parameters
        ----------
        seed: integer in range [1, 2^31 - 1]
            Fixed random seed (Default:	100). This option sets fixed seed value,
            which is used in all randomized algorithms if deterministic option is on.
            If deterministic is off, the seed value is ignored.
        """
        self._settings.set('seed', seed)

    def getLogLevel(self):
        """
        Get the log level threshold.

        Returns
        -------
        logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
            Minimum log level (Default:	"Info"). If this option is set,
            only messages with log level greater than or equal to the threshold are dumped into log.
        """
        return self._settings.get('logLevel')

    def setLogLevel(self, logLevel):
        """
        Set the log level threshold.

        Parameters
        ----------
        logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
            Minimum log level (Default:	"Info"). If this option is set,
            only messages with log level greater than or equal to the threshold are dumped into log.
        """
        self._settings.set('logLevel', logLevel)

    def getMaxParallel(self):
        """
        Get the maximum number of parallel threads.

        Returns
        -------
        maxParallel: positive integer or 0 (auto)
            Maximum number of parallel threads (Default: 0, auto).
            GTDoE can run in parallel to speed up design generation.
            This option sets the maximum number of threads it is allowed to create.
            Default setting (0) uses the value given by the OMP_NUM_THREADS environment variable,
            which by default is equal to the number of virtual processors, including hyperthreading CPUs.
        """
        return self._settings.get('maxParallel')

    def setMaxParallel(self, maxParallel):
        """
        Set the maximum number of parallel threads.

        Parameters
        ----------
        maxParallel: positive integer or 0 (auto)
            Maximum number of parallel threads (Default: 0, auto).
            GTDoE can run in parallel to speed up design generation.
            This option sets the maximum number of threads it is allowed to create.
            Default setting (0) uses the value given by the OMP_NUM_THREADS environment variable,
            which by default is equal to the number of virtual processors, including hyperthreading CPUs.
        """
        self._settings.set('maxParallel', maxParallel)


class Sequence(_Batch):
    """
    Sequential design of experiments.

    Parameters
    ----------
    bounds: tuple(list[float], list[float])
        Design space bounds (lower, upper).
    count: int, long
        The number of points to generate
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default:	100). This option sets fixed seed value,
        which is used in all randomized algorithms if deterministic option is on.
        If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default:	"Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0 (auto)
        Maximum number of parallel threads (Default: 0, auto). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create.
        Default setting (0) uses the value given by the OMP_NUM_THREADS environment variable,
        which by default is equal to the number of virtual processors, including hyperthreading CPUs.
    leap: integer in range [0,65535]
        Sequence leap size (Default: 0).
    skip: integer in range [0,65535]
        The number of elements to skip at the beginning of sequence (Default: 0).
    technique: "RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq"
        The generation algorithm to use (Default: "RandomSeq").
    """
    def __init__(self, bounds, count,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 leap=None, skip=None, technique=None):

        self.__techniques = ["RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq"]
        self.__default_technique = "SobolSeq"
        if technique in self.__techniques:
            self.__technique = technique
        else:
            self.__technique = self.__default_technique
        super(Sequence, self).__init__(bounds=bounds, count=count, technique=self.__technique)
        self._settings.set_all(deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               leap=leap, skip=skip)

    def getLeap(self):
        """
        Get the sequence leap size.

        Returns
        -------
        leap: integer in range [0,65535]
            This option allows sequential techniques to leap over elements of sequence.
            Its value is the leap size (number of elements). Default is no leaping (Default: 0).
        """
        return self._settings.get('leap')

    def setLeap(self, leap):
        """
        Set the sequence leap size.

        Parameters
        ----------
        leap: integer in range [0,65535]
            This option allows sequential techniques to leap over elements of sequence.
            Its value is the leap size (number of elements). Default is no leaping (Default: 0).
        """
        self._settings.set('leap', leap)

    def getSkip(self):
        """
        Get the number of elements to skip at the beginning of sequence.

        Returns
        -------
        skip: integer in range [0,65535]
            This option specifies the number of elements to skip from sequence start (Default: 0).
        """
        return self._settings.get('skip')

    def setSkip(self, skip):
        """
        Set the number of elements to skip at the beginning of sequence.

        Parameters
        ----------
        skip: integer in range [0,65535]
            This option specifies the number of elements to skip from sequence start (Default: 0).
        """
        self._settings.set('skip', skip)

    def getTechnique(self):
        """
        Get the generation algorithm name.

        Returns
        -------
        technique: "RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq"
            The generation algorithm to use (Default: "RandomSeq").
        """
        return self.__technique

    def setTechnique(self, technique):
        """
        Set the generation algorithm name.

        Parameters
        ----------
        technique: "RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq"
            The generation algorithm to use (Default: "RandomSeq").
        """
        if technique in self.__techniques:
            self.__technique = technique
        else:
            self.__technique = self.__default_technique
        self._settings.set('technique', self.__technique)


class LHS(_Batch):
    """
    Latin Hypercube Sampling (LHS) design of experiments.

    Parameters
    ----------
    bounds: tuple(list[float], list[float])
        Design space bounds (lower, upper).
    count: int, long
        The number of points to generate (not for the blackbox-based
        adaptive DoE; use budget).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default:	100). This option sets fixed seed value,
        which is used in all randomized algorithms if deterministic option is on.
        If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default:	"Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0 (auto)
        Maximum number of parallel threads (Default: 0, auto). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create.
        Default setting (0) uses the value given by the OMP_NUM_THREADS environment variable,
        which by default is equal to the number of virtual processors, including hyperthreading CPUs.
    useOptimized: boolean
        Use Optimized Latin Hypercube Sampling (Defalut: False)
    iterations: integer in range [2,65535]
        Maximum number of optimization iterations in OLHS generation (Default: 300).
    """
    def __init__(self, bounds, count,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 useOptimized=False, iterations=None):
        self.__use_optimized = useOptimized
        if useOptimized:
            technique = "OLHS"
        else:
            technique = "LHS"
        super(LHS, self).__init__(bounds=bounds, count=count, technique=technique)
        self._settings.set_all(deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               iterations=iterations)

    def enableOptimized(self):
        """Enable Optimized Latin Hypercube Sampling."""
        self.__use_optimized = True
        self._settings.set('technique', "OLHS")

    def disableOptimized(self):
        """Disable Optimized Latin Hypercube Sampling."""
        self.__use_optimized = False
        self._settings.set('technique', "LHS")

    def isOptimizedEnabled(self):
        """
        Test whether the Optimized Latin Hypercube Sampling is enabled or not.

        Returns
        -------
        useOptimized : bool
            Flag telling whether the Optimized Latin Hypercube Sampling is enabled.
            It is disabled by default.
        """
        return self.__use_optimized

    def getIterations(self):
        """
        Get the maximum number of optimization iterations.

        Returns
        -------
        iterations: integer in range [0,65535]
            This option allows user to specify maximum number of optimization iterations (Default: 300).
        """
        return self._settings.get('iterations')

    def setIterations(self, iterations):
        """
        Set the maximum number of optimization iterations.

        Parameters
        ----------
        iterations: integer in range [0,65535]
            This option allows user to specify maximum number of optimization iterations (Default: 300).
        """
        self._settings.set('iterations', iterations)


class Adaptive(_Batch):
    """
    Adaptive Blackbox-Based design of experiments.

    Parameters
    ----------
    blackbox: :class:`~openturns.NumericalMathFunction`
        Adaptive DoE blackbox.
    bounds: tuple(list[float], list[float])
        Design space bounds (lower, upper).
    budget: int, long
        Blackbox budget
    init_x: (array-like, 1D or 2D)
        Optional initial sample for the adaptive DoE, input part
        (values of variables).
    init_y: (array-like, 1D or 2D)
        Optional initial sample for the adaptive DoE, response part
        (function values).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default:	100). This option sets fixed seed value,
        which is used in all randomized algorithms if deterministic option is on.
        If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default:	"Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0 (auto)
        Maximum number of parallel threads (Default: 0, auto). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create.
        Default setting (0) uses the value given by the OMP_NUM_THREADS environment variable,
        which by default is equal to the number of virtual processors, including hyperthreading CPUs.
    initialDoeTechnique: "RandomSeq", "FaureSeq", "HaltonSeq", "SobolSeq", "BoxBehnken", "FullFactorial",
                         "LHS", "OLHS", "OptimalDesign", or "ParametricStudy"
        DoE technique used to generate an initial sample in the adaptive mode (Default: "LHS").
    useGradient: boolean
        Enable or disable using analytical gradients (Default: False).
    """
    def __init__(self, blackbox, bounds, budget, init_x=None, init_y=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 initialDoeTechnique=None, useGradient=False):
        super(Adaptive, self).__init__(bounds=bounds, count=None, technique=None)
        self.__use_gradient = useGradient
        self.__blackbox = blackbox
        p7_blackbox = _Blackbox(blackbox, self._settings.get('bounds'), useGradient)
        self._settings.set_all(blackbox=p7_blackbox, budget=budget,
                               init_x=init_x, init_y=init_y,
                               deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               initialDoeTechnique=initialDoeTechnique)

    def getBlackbox(self):
        """
        Get adaptive DoE blackbox

        Returns
        -------
        blackbox: :class:`~openturns.NumericalMathFunction`
            adaptive DoE blackbox, optional.
        """
        return self.__blackbox

    def setBlackbox(self, blackbox):
        """
        Set adaptive DoE blackbox

        Parameters
        -------
        blackbox: :class:`~openturns.NumericalMathFunction`
            adaptive DoE blackbox, optional.
        """
        # Convert blackbox for p7 generator
        p7_blackbox = _Blackbox(blackbox, self._settings.get('bounds'), self.__use_gradient)
        self._settings.set('blackbox', p7_blackbox)
        self.__blackbox = blackbox

    def getBudget(self):
        """
        Get blackbox budget

        Returns
        -------
        budget: int, long
            blackbox budget.
        """
        return self._settings.get('budget')

    def setBudget(srlf, budget):
        """
        Set blackbox budget

        Parameters
        -------
        budget: int, long
            blackbox budget
        """
        self._settings.set('budget', budget)

    def getInitX(self):
        """
        Get the input part of initial sample for the adaptive DoE.

        Returns
        -------
        init_x: (array-like, 1D or 2D)
            optional initial sample for the adaptive DoE, input part
            (values of variables).
        """
        return self._settings.get('init_x')

    def setInitX(self, init_x):
        """
        Set the input part of initial sample for the adaptive DoE.

        Parameters
        ----------
        init_x: (array-like, 1D or 2D)
            optional initial sample for the adaptive DoE, input part
            (values of variables).
        """
        self._settings.set('init_x', init_x)

    def getInitY(self):
        """
        Get the response part of initial sample for the adaptive DoE.

        Returns
        -------
        init_y: (array-like, 1D or 2D)
            optional initial sample for the adaptive DoE, response part
            (function values).
        """
        return self._settings.get('init_y')

    def setInitY(self, init_y):
        """
        Set the response part of initial sample for the adaptive DoE.

        Parameters
        ----------
        init_y: (array-like, 1D or 2D)
            optional initial sample for the adaptive DoE, response part
            (function values).
        """
        self._settings.set('init_y', init_y)

    def getInitialDoeTechnique(self):
        """
        Get DoE technique used to generate an initial sample.

        Returns
        -------
        initialDoeTechnique: "RandomSeq", "FaureSeq", "HaltonSeq", "SobolSeq", "BoxBehnken", "FullFactorial",
                             "LHS", "OLHS", "OptimalDesign", or "ParametricStudy"
            DoE technique used to generate an initial sample in the adaptive mode (Default: "LHS").
        """
        return self._settings.get('initialDoeTechnique')

    def setInitialDoeTechnique(self, initialDoeTechnique):
        """
        Set DoE technique used to generate an initial sample.

        Parameters
        ----------
        initialDoeTechnique: "RandomSeq", "FaureSeq", "HaltonSeq", "SobolSeq", "BoxBehnken", "FullFactorial",
                             "LHS", "OLHS", "OptimalDesign", or "ParametricStudy"
            DoE technique used to generate an initial sample in the adaptive mode (Default: "LHS").
        """
        self._settings.set('initialDoeTechnique', initialDoeTechnique)

    def enableGradient(self):
        """Enable using analytical gradients."""
        self.__use_gradient = True

    def disableGradient(self):
        """Disable using analytical gradients."""
        self.__use_gradient = False

    def isGradientEnabled(self):
        """
        Test whether the gradients are enabled or not.

        Returns
        -------
        useGradient : bool
            Flag telling whether the analytical gradients are enabled.
            It is disabled by default.
        """
        return self.__use_gradient