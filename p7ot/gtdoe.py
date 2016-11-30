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

import numpy as np
import openturns as ot
from da.p7core import gtdoe

from .blackbox import _Blackbox


# Used to keep DoE settings in appropriate way (for p7core.gtdoe generator)
class _DoeSettings(object):

    # {variable_name: option_name} dictionary
    # Each option corresponds to a variable.
    __SETTINGS_TO_OPTIONS = {
        # Basic DoE options
        'deterministic': 'GTDoE/Deterministic',
        'logLevel': 'GTDoE/LogLevel',
        'seed': 'GTDoE/Seed',
        # Advanced DoE options
        'categoricalVariables': 'GTDoE/CategoricalVariables',
        'maxParallel': 'GTDoE/MaxParallel',
        'technique': 'GTDoE/Technique',
        # Optimized Latin Hypercube Sampling DoE options
        'iterations': 'GTDoE/OLHS/Iterations',
        # Sequential DoE options
        'leap': 'GTDoE/Sequential/Leap',
        'skip': 'GTDoE/Sequential/Skip',
        # Fractional Factorial DoE options
        'generatingString': 'GTDoE/FractionalFactorial/GeneratingString',
        'mainFactors': 'GTDoE/FractionalFactorial/MainFactors',
        # Orthogonal Array DoE options
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
        # Box-Behnken DoE options
        'isFull': 'GTDoE/BoxBehnken/IsFull',
        # OptimalDesign DoE options
        'model': 'GTDoE/OptimalDesign/Model',
        'tries': 'GTDoE/OptimalDesign/Tries',
        'criterionType': 'GTDoE/OptimalDesign/Type'
    }

    def __init__(self):
        self.__settings = {'options': {}}
        self.__options = gtdoe.Generator().options

    def set_all(self, **kwargs):
        for key, value in kwargs.items():
            self.set(key=key, value=value)

    def set(self, key, value):
        option = self.__SETTINGS_TO_OPTIONS.get(key)
        if option:
            self.__options.set(option, value)
        else:
            self.__settings[key] = value

    def get(self, key=None):
        if key is None:
            # return all settings (dictionary)
            self.__settings['options'] = self.__options.get()
            return self.__settings
        option = self.__SETTINGS_TO_OPTIONS.get(key)
        if option:
            return self.__options.get(option)
        else:
            return self.__settings.get(key)


class _Experiment(object):
    """
    Base class for da.p7core.gtdoe techniques.

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long
        The number of points to generate.
    """

    def __init__(self, bounds, count):
        self.__generator = gtdoe.Generator()
        self.__p7_result = None
        self.__bounds = None
        # preparing the parameters of p7core.gtdoe generate method
        self._settings = _DoeSettings()
        self.setBounds(bounds)
        self.setCount(count)
        super(_Experiment, self).__init__()

    def generate(self):
        """
        Generate points according to the experiment settings.

        Returns
        -------
        sample: :class:`~openturns.NumericalSample`
            The points which constitute the design of experiments.
        """
        self.__p7_result = self.__generator.generate(**self._settings.get())
        return ot.NumericalSample(self.__p7_result.points)

    def getClassName(self):
        """
        Accessor to the object's name.

        Returns
        -------
        class_name: str
            The object class name (`object.__class__.__name__`).
        """
        return self.__class__.__name__

    def getBounds(self):
        """
        Get design space bounds.

        Returns
        -------
        bounds: :class:`~openturns.Interval`
            Design space bounds.
        """
        return self.__bounds

    def setBounds(self, bounds):
        """
        Set design space bounds.

        Parameters
        -------
        bounds: :class:`~openturns.Interval`
            Design space bounds.
        """
        if not isinstance(bounds, ot.Interval):
            raise TypeError("Wrong 'bounds' type %s! Required: %s" % (type(bounds).__name__, ot.Interval))
        self.__bounds = bounds
        # Convert bounds for p7 generator
        p7_bounds = (list(bounds.getLowerBound()), list(bounds.getUpperBound()))
        self._settings.set('bounds', p7_bounds)

    def getCount(self):
        """
        Get the number of points to generate.

        Returns
        -------
        count: int, long
            The number of points to generate.
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
        if not isinstance(count, (int, long)):
            raise TypeError("Wrong 'count' type %s! Required: int or long" % type(count).__name__)
        if count <= 0:
            raise ValueError("Argument 'count' must be > 0")
        self._settings.set('count', count)

    def setLogger(self, logger):
        """
        Set up a logger for the DoE generation process.

        Parameters
        ----------
        logger: da.p7core.loggers.StreamLogger object
            See section Loggers in p7core documentation for details.
        """
        self.__generator.set_logger(logger)

    def setWatcher(self, watcher):
        """
        Set up a watcher for the DoE generation process.

        Parameters
        ----------
        watcher: da.p7core.watchers.DefaultWatcher object
            A watcher is an object that is capable of interrupting a process.
            See section Watchers in p7core documentation for details.
        """
        self.__generator.set_watcher(watcher)

    def getDeterministic(self):
        """
        Test whether the random seed is fixed or not.

        Returns
        -------
        deterministic: boolean
            Require generation to be reproducible (Default: False).
        """
        return self._settings.get('deterministic')

    def setDeterministic(self, deterministic):
        """
        Tell whether the random seed is fixed or not.

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
            Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
            algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
        """

        return self._settings.get('seed')

    def setSeed(self, seed):
        """
        Set the value of fixed random seed.

        Parameters
        ----------
        seed: integer in range [1, 2^31 - 1]
            Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
            algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
        """
        self._settings.set('seed', seed)

    def getLogLevel(self):
        """
        Get the log level threshold.

        Returns
        -------
        logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
            Minimum log level (Default: "Info"). If this option is set,
            only messages with log level greater than or equal to the threshold are dumped into log.
        """
        return self._settings.get('logLevel')

    def setLogLevel(self, logLevel):
        """
        Set the log level threshold.

        Parameters
        ----------
        logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
            Minimum log level (Default: "Info"). If this option is set,
            only messages with log level greater than or equal to the threshold are dumped into log.
        """
        self._settings.set('logLevel', logLevel)

    def getCategoricalVariables(self):
        """
        Get the categorical variables.

        Returns
        -------
        categoricalVariables: a list in JSON format
            Specifies the indices of categorical variables and their categories for the GTDoE generator.
            Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
            index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
            and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
        """
        return self._settings.get('categoricalVariables')

    def setCategoricalVariables(self, categoricalVariables):
        """
        Set the categorical variables.

        Parameters
        ----------
        categoricalVariables: a list in JSON format
            Specifies the indices of categorical variables and their categories for the GTDoE generator.
            Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
            index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
            and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
        """
        self._settings.set('categoricalVariables', categoricalVariables)

    def getMaxParallel(self):
        """
        Get the maximum number of parallel threads.

        Returns
        -------
        maxParallel: positive integer or 0
            Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
            This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
            given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
            processors, including hyperthreading CPUs.
        """
        return self._settings.get('maxParallel')

    def setMaxParallel(self, maxParallel):
        """
        Set the maximum number of parallel threads.

        Parameters
        ----------
        maxParallel: positive integer or 0
            Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
            This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
            given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
            processors, including hyperthreading CPUs.
        """
        self._settings.set('maxParallel', maxParallel)

    def getP7Result(self):
        """
        Accessor to p7core DoE result.

        Returns
        -------
        p7result: :class:`~da.p7core.gtdoe.Result`
            Result object.
        """
        return self.__p7_result


class Sequence(_Experiment):
    """
    Sequential design of experiments.

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long
        The number of points to generate.
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    leap: integer in range [0, 65535]
        This option allows sequential techniques to leap over elements of sequence.
        Its value is the leap size (number of elements). Default is no leaping (Default: 0).
    skip: integer in range [0, 65535]
        This option specifies the number of elements to skip from sequence start (Default: 0).
    technique: "RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq"
        The generation algorithm to use (Default: "SobolSeq").
    """

    def __init__(self, bounds, count,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 leap=None, skip=None, technique=None):
        super(Sequence, self).__init__(bounds=bounds, count=count)
        self.setTechnique(technique)
        self._settings.set_all(deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               leap=leap, skip=skip)

    def getLeap(self):
        """
        Get the sequence leap size.

        Returns
        -------
        leap: integer in range [0, 65535]
            This option allows sequential techniques to leap over elements of sequence.
            Its value is the leap size (number of elements). Default is no leaping (Default: 0).
        """
        return self._settings.get('leap')

    def setLeap(self, leap):
        """
        Set the sequence leap size.

        Parameters
        ----------
        leap: integer in range [0, 65535]
            This option allows sequential techniques to leap over elements of sequence.
            Its value is the leap size (number of elements). Default is no leaping (Default: 0).
        """
        self._settings.set('leap', leap)

    def getSkip(self):
        """
        Get the number of elements to skip at the beginning of sequence.

        Returns
        -------
        skip: integer in range [0, 65535]
            This option specifies the number of elements to skip from sequence start (Default: 0).
        """
        return self._settings.get('skip')

    def setSkip(self, skip):
        """
        Set the number of elements to skip at the beginning of sequence.

        Parameters
        ----------
        skip: integer in range [0, 65535]
            This option specifies the number of elements to skip from sequence start (Default: 0).
        """
        self._settings.set('skip', skip)

    def getTechnique(self):
        """
        Get the generation algorithm name.

        Returns
        -------
        technique: "RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq"
            The generation algorithm to use (Default: "SobolSeq").
        """
        return self._settings.get('technique')

    def setTechnique(self, technique):
        """
        Set the generation algorithm name.

        Parameters
        ----------
        technique: "RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq"
            The generation algorithm to use (Default: "SobolSeq").
        """
        if technique is None:
            self._settings.set('technique', "SobolSeq")
        else:
            if not isinstance(technique, str):
                raise TypeError("Wrong 'technique' type %s! Required: str" % type(technique).__name__)
            if technique not in ["RandomSeq", "SobolSeq", "HaltonSeq", "FaureSeq"]:
                raise ValueError("Unknown 'technique' value '%s'." % technique +
                                 " Expected values are: 'RandomSeq', 'HaltonSeq', 'SobolSeq', 'FaureSeq'")
            self._settings.set('technique', technique)


class LHS(_Experiment):
    """
    Latin Hypercube Sampling (LHS) design of experiments.

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long
        The number of points to generate.
    categoricalVariables: a list in JSON format
        Specifies the indices of categorical variables and their categories for the GTDoE generator.
        Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
        index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
        and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    useOptimized: boolean
        Use Optimized Latin Hypercube Sampling (Default: False).
    iterations: integer in range [2, 65535]
        Maximum number of optimization iterations in OLHS generation (Default: 300).
    """

    def __init__(self, bounds, count,
                 categoricalVariables=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 useOptimized=False, iterations=None):
        super(LHS, self).__init__(bounds=bounds, count=count)
        if not isinstance(useOptimized, bool):
            raise TypeError("Wrong 'useOptimized' type %s! Required: bool" % type(useOptimized).__name__)
        self.__use_optimized = useOptimized
        technique = "OLHS" if useOptimized else "LHS"
        self._settings.set_all(categoricalVariables=categoricalVariables,
                               deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               iterations=iterations, technique=technique)

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
        useOptimized: bool
            Flag telling whether the Optimized Latin Hypercube Sampling is enabled (Default: False).
        """
        return self.__use_optimized

    def getIterations(self):
        """
        Get the maximum number of optimization iterations.

        Returns
        -------
        iterations: integer in range [0, 65535]
            This option allows user to specify maximum number of optimization iterations (Default: 300).
        """
        return self._settings.get('iterations')

    def setIterations(self, iterations):
        """
        Set the maximum number of optimization iterations.

        Parameters
        ----------
        iterations: integer in range [0, 65535]
            This option allows user to specify maximum number of optimization iterations (Default: 300).
        """
        self._settings.set('iterations', iterations)


class BoxBehnken(_Experiment):
    """
    Box-Behnken design of experiments.

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long in range [1, 2d(d-1)+1]
        The number of points to generate (Default: None). If this option is None, GTDoE will generate a full Box-Behnken
        design including 2d(d-1)+1 points where d is the number of design variables. Box-Behnken design generation
        respects the number of points to generate by randomly excluding some points from a full design to return the
        requested number of points in case the latter is less than the number of points in the full design (2d(d-1)+1).
        Note that the full design is the maximum sample size which can be generated by the Box-Behnken technique,
        so if the requested number of points exceeds 2d(d-1)+1, point generation will not start.
    categoricalVariables: a list in JSON format
        Specifies the indices of categorical variables and their categories for the GTDoE generator.
        Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
        index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
        and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    """

    def __init__(self, bounds, count=None,
                 categoricalVariables=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None):
        super(BoxBehnken, self).__init__(bounds=bounds, count=count)
        self._settings.set_all(categoricalVariables=categoricalVariables,
                               deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               technique='BoxBehnken')

    def setCount(self, count):
        """
        Set the number of points to generate.

        Parameters
        ----------
        count: int, long
            The number of points to generate.
        """
        if count is None:
            d = self.getBounds().getDimension()
            self._settings.set('isFull', True)
            self._settings.set('count', 2 * d * (d - 1) + 1)
        else:
            super(BoxBehnken, self).setCount(count)


class FullFactorial(_Experiment):
    """
    Full factorial design of experiments (uniform mesh).

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long greater than or equal to 2^dim
        The number of points to generate. Actual number of points to be generated: int(count**(1.0/dim))**dim
    categoricalVariables: a list in JSON format
        Specifies the indices of categorical variables and their categories for the GTDoE generator.
        Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
        index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
        and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    """

    def __init__(self, bounds, count,
                 categoricalVariables=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None):
        super(FullFactorial, self).__init__(bounds=bounds, count=count)
        self._settings.set_all(categoricalVariables=categoricalVariables,
                               deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               technique='FullFactorial')


class FractionalFactorial(_Experiment):
    """
    Fractional factorial design of experiments.

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long less than or equal to 2^dim
        The number of points to generate.
    categoricalVariables: a list in JSON format
        Specifies the indices of categorical variables and their categories for the GTDoE generator.
        Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
        index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
        and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
        Also note that when this option is used with the Fractional Factorial technique, each categorical
        variable must have exactly two categories. For example, [0, [2., 3.], 4, [0.1, 0.2, 0.3]] specifies that
        two of the design variables (indexed 0 and 4) are categorical, and defines the categories for each.
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    generatingString: a string containing generator expressions, separated by whitespace
        Specifies alias structure for a fractional factorial design (Default: "").
        This option uses the conventional fractional factorial notation to create an alias structure that determines
        which effects are confounded with each other. Each generator expression contains one or more letters.
        Single letter expression specifies a main factor, letter combinations give interactions for confound
        factors. For example, "a b c ab bcd d" means that variables indexed 0, 1, 2, and 5 are main factors and
        a full factorial design for them is generated; design values for variables 3 and 4 are generated from main
        factor values: for each point, the value of variable 3 is the product of variables 0 and 1 (“ab”),
        and the value of variable 4 is the product of variables 1, 2, and 3 (“bcd”).
    mainFactors: list of indices of main factors (variables)
        Specifies main (independent) design factors (Default: []).
        This option provides a simplified way to create an alias structure for a fractional factorial design
        (compared with generatingString). The list contains only the indices of variables to be selected as main
        factors, and interactions for confound factors are then created automatically. If both generatingString and
        mainFactors are specified, they must be consistent (that is, select the same main factors). If both these
        options are left default, GTDoE first selects a number of main factors so it is enough to generate
        the requested number of points, then adds generator expressions for the remaining factors.
    Notes
    -----
    A design consisting of a subset (fraction) of a full factorial design.
    """

    def __init__(self, bounds, count,
                 categoricalVariables=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 generatingString=None, mainFactors=None):
        super(FractionalFactorial, self).__init__(bounds=bounds, count=count)
        self._settings.set_all(categoricalVariables=categoricalVariables,
                               deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               generatingString=generatingString, mainFactors=mainFactors,
                               technique='FractionalFactorial')

    def getGeneratingString(self):
        """
        Get the alias structure for a fractional factorial design.

        Returns
        -------
        generatingString: a string containing generator expressions, separated by whitespace
            Specifies alias structure for a fractional factorial design (Default: "").
            This option uses the conventional fractional factorial notation to create an alias structure that determines
            which effects are confounded with each other. Each generator expression contains one or more letters.
            Single letter expression specifies a main factor, letter combinations give interactions for confound
            factors. For example, "a b c ab bcd d" means that variables indexed 0, 1, 2, and 5 are main factors and
            a full factorial design for them is generated; design values for variables 3 and 4 are generated from main
            factor values: for each point, the value of variable 3 is the product of variables 0 and 1 (“ab”),
            and the value of variable 4 is the product of variables 1, 2, and 3 (“bcd”).
        """
        return self._settings.get('generatingString')

    def setGeneratingString(self, generatingString):
        """
        Set the alias structure for a fractional factorial design.

        Parameters
        ----------
        generatingString: a string containing generator expressions, separated by whitespace
            Specifies alias structure for a fractional factorial design (Default: "").
            This option uses the conventional fractional factorial notation to create an alias structure that determines
            which effects are confounded with each other. Each generator expression contains one or more letters.
            Single letter expression specifies a main factor, letter combinations give interactions for confound
            factors. For example, "a b c ab bcd d" means that variables indexed 0, 1, 2, and 5 are main factors and
            a full factorial design for them is generated; design values for variables 3 and 4 are generated from main
            factor values: for each point, the value of variable 3 is the product of variables 0 and 1 (“ab”),
            and the value of variable 4 is the product of variables 1, 2, and 3 (“bcd”).
        """
        self._settings.set('generatingString', generatingString)

    def getMainFactors(self):
        """
        Get the main (independent) design factors.

        Returns
        -------
        mainFactors: list of indices of main factors (variables)
            Specifies main (independent) design factors (Default: []).
            This option provides a simplified way to create an alias structure for a fractional factorial design
            (compared with generatingString). The list contains only the indices of variables to be selected as main
            factors, and interactions for confound factors are then created automatically. If both generatingString and
            mainFactors are specified, they must be consistent (that is, select the same main factors). If both these
            options are left default, GTDoE first selects a number of main factors so it is enough to generate
            the requested number of points, then adds generator expressions for the remaining factors.
        """
        return self._settings.get('mainFactors')

    def setMainFactors(self, mainFactors):
        """
        Set the main (independent) design factors.

        Parameters
        ----------
        mainFactors: list of indices of main factors (variables)
            Specifies main (independent) design factors (Default: []).
            This option provides a simplified way to create an alias structure for a fractional factorial design
            (compared with generatingString). The list contains only the indices of variables to be selected as main
            factors, and interactions for confound factors are then created automatically. If both generatingString and
            mainFactors are specified, they must be consistent (that is, select the same main factors). If both these
            options are left default, GTDoE first selects a number of main factors so it is enough to generate
            the requested number of points, then adds generator expressions for the remaining factors.
        """
        self._settings.set('mainFactors', mainFactors)


class OptimalDesign(_Experiment):
    """
    Optimal design of experiments for response surface models.

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long
        The number of points to generate.
    categoricalVariables: a list in JSON format
        Specifies the indices of categorical variables and their categories for the GTDoE generator.
        Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
        index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
        and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    model: "linear", "interaction", "quadratic", or "purequadratic" (Default: "linear")
        The type of the regression model to optimize for. This option controls the order of the regression model.
        "linear" - model includes constant and linear terms. "interaction" - model includes constant, linear, and
        cross product terms. "quadratic" - model includes constant, linear, cross product and squared terms.
        "purequadratic" - model includes constant, linear and squared terms.
    tries: integer in range [1, 2^32-1]
        The number of optimal design generation tries. Sets maximum number of tries to generate a design from new
        starting point, using random points for each try (Default: 1).
    criterionType: "D" (D-optimal) or "I" (I-optimal)  (Default: "D")
        The type of optimality criterion. Specifies the type of objective function to evaluate experimental design.
        D-optimality (determinant) criterion results in maximizing the differential Shannon information content of
        the parameter estimates. I-optimality (integrated): seeks to minimize the average prediction variance over
        the design space.
    """

    def __init__(self, bounds, count,
                 categoricalVariables=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 model=None, tries=None, criterionType=None):
        super(OptimalDesign, self).__init__(bounds=bounds, count=count)
        self._settings.set_all(categoricalVariables=categoricalVariables,
                               deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               model=model, tries=tries, criterionType=criterionType,
                               technique='OptimalDesign')

    def getModel(self):
        """
        Get the type of the regression model.

        Returns
        -------
        model: "linear", "interaction", "quadratic", or "purequadratic" (Default: "linear")
            The type of the regression model to optimize for. This option controls the order of the regression model.
            "linear" - model includes constant and linear terms. "interaction" - model includes constant, linear, and
            cross product terms. "quadratic" - model includes constant, linear, cross product and squared terms.
            "purequadratic" - model includes constant, linear and squared terms.
        """
        return self._settings.get('model')

    def setModel(self, model):
        """
        Set the type of the regression model.

        Parameters
        ----------
        model: "linear", "interaction", "quadratic", or "purequadratic" (Default: "linear")
            The type of the regression model to optimize for. This option controls the order of the regression model.
            "linear" - model includes constant and linear terms. "interaction" - model includes constant, linear, and
            cross product terms. "quadratic" - model includes constant, linear, cross product and squared terms.
            "purequadratic" - model includes constant, linear and squared terms.
        """
        self._settings.set('model', model)

    def getTries(self):
        """
        Get the number of optimal design generation tries.

        Returns
        -------
        tries: integer in range [1, 2^32-1]
            The number of optimal design generation tries. Sets maximum number of tries to generate a design from new
            starting point, using random points for each try (Default: 1).
        """
        return self._settings.get('tries')

    def setTries(self, tries):
        """
        Set the number of optimal design generation tries.

        Parameters
        ----------
        tries: integer in range [1, 2^32-1]
            The number of optimal design generation tries. Sets maximum number of tries to generate a design from new
            starting point, using random points for each try (Default: 1).
        """
        self._settings.set('tries', tries)

    def getCriterionType(self):
        """
        Get the optimality criterion.

        Returns
        -------
        criterionType: "D" (D-optimal) or "I" (I-optimal) (Default: "D")
            The type of optimality criterion. Specifies the type of objective function to evaluate experimental design.
            D-optimality (determinant) criterion results in maximizing the differential Shannon information content of
            the parameter estimates. I-optimality (integrated): seeks to minimize the average prediction variance over
            the design space.
        """
        return self._settings.get('criterionType')

    def setCriterionType(self, criterionType):
        """
        Set the optimality criterion.

        Parameters
        ----------
        criterionType: "D" (D-optimal) or "I" (I-optimal) (Default: "D")
            The type of optimality criterion. Specifies the type of objective function to evaluate experimental design.
            D-optimality (determinant) criterion results in maximizing the differential Shannon information content of
            the parameter estimates. I-optimality (integrated): seeks to minimize the average prediction variance over
            the design space.
        """
        self._settings.set('criterionType', criterionType)


class OrthogonalArray(_Experiment):
    """
    Design of experiments with multilevel discrete design variables.

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    levelsNumber: list of indices of main factors (variables)
        Specifies levels for each factor required. Array with the number of levels for each factor
        of the orthogonal array. It should contain the same number of elements as the number of factors.
        Each element should be an integer which is greater or equal than two.
    categoricalVariables: a list in JSON format
        Specifies the indices of categorical variables and their categories for the GTDoE generator.
        Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
        index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
        and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    maxIterations: integer in range [1, 1000]
        Maximum number of iterations per dimension for greedy search (Default: 10).
    multistartIterations: integer in range [1, 1000]
        Number of algorithm multistart iterations (Default: 10).
    """

    def __init__(self, bounds, levelsNumber,
                 categoricalVariables=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 maxIterations=None, multistartIterations=None):
        super(OrthogonalArray, self).__init__(bounds=bounds, count=levelsNumber)
        self._settings.set_all(categoricalVariables=categoricalVariables,
                               deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               maxIterations=maxIterations, multistartIterations=multistartIterations,
                               technique='OrthogonalArray')

    def setCount(self, levelsNumber):
        """
        Set the number of points to generate.

        Parameters
        ----------
        levelsNumber: list of indices of main factors (variables)
            Specifies levels for each factor, required. Array with the number of levels for each factor
            of the orthogonal array. It should contain the same number of elements as the number of factors.
            Each element should be an integer which is greater or equal than two.
        """
        self._settings.set("levelsNumber", levelsNumber)
        count = 1
        for value in levelsNumber:
            count *= value
        self._settings.set('count', count)

    def getLevelsNumber(self):
        """
        Get the levels for each factor.

        Returns
        -------
        levelsNumber: list of indices of main factors (variables)
            Specifies levels for each factor, required. Array with the number of levels for each factor
            of the orthogonal array. It should contain the same number of elements as the number of factors.
            Each element should be an integer which is greater or equal than two.
        """
        return self._settings.get('levelsNumber')

    def setLevelsNumber(self, levelsNumber):
        """
        Set the levels for each factor.

        Parameters
        ----------
        levelsNumber: list of indices of main factors (variables)
            Specifies levels for each factor, required. Array with the number of levels for each factor
            of the orthogonal array. It should contain the same number of elements as the number of factors.
            Each element should be an integer which is greater or equal than two.
        """
        self._settings.set('levelsNumber', levelsNumber)

    def getMaxIterations(self):
        """
        Get the maximum number of iterations.

        Returns
        -------
        maxIterations: integer in range [1, 1000]
            Maximum number of iterations per dimension for greedy search (Default: 10).
        """
        return self._settings.get('maxIterations')

    def setMaxIterations(self, maxIterations):
        """
        Set the maximum number of iterations.

        Parameters
        ----------
        maxIterations: integer in range [1, 1000]
            Maximum number of iterations per dimension for greedy search (Default: 10).
        """
        self._settings.set('maxIterations', maxIterations)

    def getMultistartIterations(self):
        """
        Get the number of algorithm multistart iterations.

        Returns
        -------
        multistartIterations: integer in range [1, 1000]
            Number of algorithm multistart iterations (Default: 10).
        """
        return self._settings.get('multistartIterations')

    def setMultistartIterations(self, multistartIterations):
        """
        Set the number of algorithm multistart iterations.

        Parameters
        ----------
        multistartIterations: integer in range [1, 1000]
            Number of algorithm multistart iterations (Default: 10).
        """
        self._settings.set('multistartIterations', multistartIterations)


class ParametricStudy(_Experiment):
    """
    Parametric study process (select a central point and generate points from center by changing one component)

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long
        The number of points to generate.
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    """

    def __init__(self, bounds, count,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None):
        super(ParametricStudy, self).__init__(bounds=bounds, count=count)
        self._settings.set_all(deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               technique='ParametricStudy')


class _Adaptive(_Experiment):

    def _check_initial_sample(self, sample):
        if not isinstance(sample, (list, tuple, np.ndarray, ot.NumericalSample, ot.NumericalPoint)):
            raise TypeError("Wrong initial sample type %s! Required: array-like" % type(sample).__name__)

    def getInitX(self):
        """
        Get the input part of initial sample for the adaptive DoE.

        Returns
        -------
        init_x: (array-like, 1D or 2D)
            Initial sample for the adaptive DoE, input part (values of variables).
        """
        return self._settings.get('init_x')

    def setInitX(self, init_x):
        """
        Set the input part of initial sample for the adaptive DoE.

        Parameters
        ----------
        init_x: (array-like, 1D or 2D)
            Initial sample for the adaptive DoE, input part (values of variables).
        """
        self._check_initial_sample(init_x)
        self._settings.set('init_x', init_x)

    def getAccelerator(self):
        """
        Get the accelerator value.

        Returns
        -------
        accelerator: integer in range [1, 5]
            Five-position switch to control the trade-off between speed and accuracy of approximations
            used by adaptive DoE (Default: 1).
        """
        return self._settings.get('accelerator')

    def setAccelerator(self, accelerator):
        """
        Set the accelerator value.

        Parameters
        ----------
        accelerator: integer in range [1, 5]
            Five-position switch to control the trade-off between speed and accuracy of approximations
            used by adaptive DoE (Default: 1).
        """
        self._settings.set('accelerator', accelerator)

    def getAnnealingCount(self):
        """
        Get the number of criterion evaluations in simulated annealing procedure.

        Returns
        -------
        annealingCount: integer in range [1, 2^31-2]
            Each sample point added by adaptive DoE process is the result of optimization by selected
            adaptive DoE criterion. This optimization is iterative, and the more iterations it can make,
            the better the new point will probably be. This option adjusts the number of optimizer iterations.
            Note that it also directly affects the working time of adaptive DoE algorithm.
            Default (0) is an automatic estimate based on the design space dimensionality (d_in),
            which sets the number of iterations to min(500+100*d_in, 3000).
        """
        return self._settings.get('annealingCount')

    def setAnnealingCount(self, annealingCount):
        """
        Set the number of criterion evaluations in simulated annealing procedure.

        Parameters
        ----------
        annealingCount: integer in range [1, 2^31-2]
            Each sample point added by adaptive DoE process is the result of optimization by selected
            adaptive DoE criterion. This optimization is iterative, and the more iterations it can make,
            the better the new point will probably be. This option adjusts the number of optimizer iterations.
            Note that it also directly affects the working time of adaptive DoE algorithm.
            Default (0) is an automatic estimate based on the design space dimensionality (d_in),
            which sets the number of iterations to min(500+100*d_in, 3000).
        """
        self._settings.set('annealingCount', annealingCount)

    def getCriterion(self):
        """
        Get the criterion for placing new DoE points.

        Returns
        -------
        criterion: "IntegratedMseGainMaxVar", "MaximumVariance", "Uniform", or "Auto"
            Control the behavior of adaptive generation algorithm. "IntegratedMseGainMaxVar": most accurate and
            time-consuming method. Estimates the error of approximation with new candidate point added to the sample
            and selects the point which minimizes the expected error. "MaximumVariance": samples points in the
            region with highest uncertainty, relying on model accuracy evaluation. Faster but less accurate.
            "Uniform": does not aim to increase model quality. Generates next sample point in such a way that
            overall sample is as uniform as possible. Note that in fact this is the only valid criterion for the
            sample-based adaptive DoE when only the input part of the initial sample is available. "Auto": defaults
            to "MaximumVariance" if both input and response parts of the initial sample are available, and
            to "Uniform" if only the input part is available.
        """
        return self._settings.get('criterion')

    def setCriterion(self, criterion):
        """
        Set the criterion for placing new DoE points.

        Parameters
        ----------
        criterion: "IntegratedMseGainMaxVar", "MaximumVariance", "Uniform", or "Auto"
            Control the behavior of adaptive generation algorithm. "IntegratedMseGainMaxVar": most accurate and
            time-consuming method. Estimates the error of approximation with new candidate point added to the sample
            and selects the point which minimizes the expected error. "MaximumVariance": samples points in the
            region with highest uncertainty, relying on model accuracy evaluation. Faster but less accurate.
            "Uniform": does not aim to increase model quality. Generates next sample point in such a way that
            overall sample is as uniform as possible. Note that in fact this is the only valid criterion for the
            sample-based adaptive DoE when only the input part of the initial sample is available. "Auto": defaults
            to "MaximumVariance" if both input and response parts of the initial sample are available, and
            to "Uniform" if only the input part is available.
        """
        self._settings.set('criterion', criterion)

    def getExactFitRequired(self):
        """
        Test whether the exact fit is required or not.

        Returns
        -------
        exactFitRequired: Boolean
            Require all approximations to fit the training data exactly. If this option is on (True),
            all approximations constructed in the adaptive DoE process fit the sample points exactly.
            If off (False) then no fitting condition is imposed (Default: False).
        """
        return self._settings.get('exactFitRequired')

    def setExactFitRequired(self, exactFitRequired):
        """
        Tell whether the exact fit is required or not.

        Parameters
        ----------
        exactFitRequired: Boolean
            Require all approximations to fit the training data exactly. If this option is on (True),
            all approximations constructed in the adaptive DoE process fit the sample points exactly.
            If off (False) then no fitting condition is imposed (Default: False).
        """
        self._settings.set('exactFitRequired', exactFitRequired)

    def getInternalValidation(self):
        """
        Test whether the internal validation is enabled or not.

        Returns
        -------
        internalValidation: Boolean
            Enable or disable internal validation for approximations used by adaptive DoE. Note that internal
            validation scores are computed for every model built, including intermediate models built between DoE
            iterations, so switching this on may significantly increase DoE generation time. Note that in the
            sample-based adaptive DoE mode, internal validation requires the response part of the initial sample.
            This is due to the fact that sample-based adaptive DoE with initial input part only is just random uniform
            generation which does not involve approximation models, so the option has no sense.
        """
        return self._settings.get('internalValidation')

    def setInternalValidation(self, internalValidation):
        """
        Tell whether the internal validation is enabled or not.

        Parameters
        ----------
        internalValidation: Boolean
            Enable or disable internal validation for approximations used by adaptive DoE. Note that internal
            validation scores are computed for every model built, including intermediate models built between DoE
            iterations, so switching this on may significantly increase DoE generation time. Note that in the
            sample-based adaptive DoE mode, internal validation requires the response part of the initial sample.
            This is due to the fact that sample-based adaptive DoE with initial input part only is just random uniform
            generation which does not involve approximation models, so the option has no sense.
        """
        self._settings.set('internalValidation', internalValidation)


class AdaptiveBlackbox(_Adaptive):
    """
    Adaptive Blackbox-Based design of experiments.

    Parameters
    ----------
    blackbox: :class:`~openturns.NumericalMathFunction`
        Adaptive DoE blackbox.
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long
        The number of points to generate. It will be used to set the budget.
    categoricalVariables: a list in JSON format
        Specifies the indices of categorical variables and their categories for the GTDoE generator.
        Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
        index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
        and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    init_x: (array-like, 1D or 2D)
        Initial sample for the adaptive DoE, input part (values of variables).
    init_y: (array-like, 1D or 2D)
        Initial sample for the adaptive DoE, response part (function values).
    initialDoeTechnique: "RandomSeq", "FaureSeq", "HaltonSeq", "SobolSeq", "BoxBehnken", "FullFactorial",
                         "LHS", "OLHS", "OptimalDesign", or "ParametricStudy"
        DoE technique used to generate an initial sample in the adaptive mode (Default: "LHS").
    accelerator: integer in range [1, 5]
        Five-position switch to control the trade-off between speed and accuracy of approximations
        used by adaptive DoE (Default: 1).
    annealingCount: integer in range [1, 2^31-2]
        Each sample point added by adaptive DoE process is the result of optimization by selected
        adaptive DoE criterion. This optimization is iterative, and the more iterations it can make,
        the better the new point will probably be. This option adjusts the number of optimizer iterations.
        Note that it also directly affects the working time of adaptive DoE algorithm.
        Default (0) is an automatic estimate based on the design space dimensionality (d_in),
        which sets the number of iterations to min(500+100*d_in, 3000).
    criterion: "IntegratedMseGainMaxVar", "MaximumVariance", "Uniform", or "Auto"
        Control the behavior of adaptive generation algorithm. "IntegratedMseGainMaxVar": most accurate and
        time-consuming method. Estimates the error of approximation with new candidate point added to the sample
        and selects the point which minimizes the expected error. "MaximumVariance": samples points in the
        region with highest uncertainty, relying on model accuracy evaluation. Faster but less accurate.
        "Uniform": does not aim to increase model quality. Generates next sample point in such a way that
        overall sample is as uniform as possible. Note that in fact this is the only valid criterion for the
        sample-based adaptive DoE when only the input part of the initial sample is available. "Auto": defaults
        to "MaximumVariance" if both input and response parts of the initial sample are available, and
        to "Uniform" if only the input part is available.
    exactFitRequired: Boolean
        Require all approximations to fit the training data exactly. If this option is on (True),
        all approximations constructed in the adaptive DoE process fit the sample points exactly.
        If off (False) then no fitting condition is imposed (Default: False).
    internalValidation: Boolean
        Enable or disable internal validation for approximations used by adaptive DoE. Note that internal
        validation scores are computed for every model built, including intermediate models built between DoE
        iterations, so switching this on may significantly increase DoE generation time. Note that in the
        sample-based adaptive DoE mode, internal validation requires the response part of the initial sample.
        This is due to the fact that sample-based adaptive DoE with initial input part only is just random uniform
        generation which does not involve approximation models, so the option has no sense.
    initialCount: 0, or an integer in range [2*d_in+3, count] (except when initialDoeTechnique is "FullFactorial"),
                  where d_in is the design space dimensionality.
        In case initial training set was not provided by user, the tool generates a sample using the technique
        specified by the initialDoeTechnique option (Latin hypercube sampling by default). This sample is then
        evaluated with the blackbox, and the generated inputs and obtained blackbox outputs are used as an
        initial training set. initialCount sets the size of this sample. If left default, the size will be
        automatically set equal to 2*d_in+3, where d_in is the design space dimensionality (Default: 0).
    oneStepCount: integer in range [1, 2^31-2]
        Each adaptive DoE step may generate more than one point. This option sets the amount of points requested
        on each step. Note that it is not always possible to generate requested number of points; in such case
        maximum possible number of points is generated. This option takes effect in the blackbox-based adaptive
        DoE mode only. Sample-based adaptive DoE disregards this option (Default: 1).
    trainIterations: integer in range [1, 2^31-2]
        Usually the approximation model used by adaptive DoE process is expected to only change slightly when a
        few points are added to the training set, so there is no need to rebuild the model at every step. This
        assumption, however, is not always true, especially when training set is not big enough to ensure that
        approximation has reasonable quality. This option sets the number of steps between rebuilds.
    """

    def __init__(self, blackbox, bounds, count,
                 categoricalVariables=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 init_x=None, init_y=None, initialDoeTechnique=None,
                 accelerator=None, annealingCount=None, criterion=None,
                 exactFitRequired=None, internalValidation=None,
                 initialCount=None, oneStepCount=None, trainIterations=None):
        super(AdaptiveBlackbox, self).__init__(bounds=bounds, count=count)
        init_x = [] if init_x is None else init_x
        init_y = [] if init_y is None else init_y
        self.__p7_blackbox = None
        self.__blackbox = None
        self.setBlackbox(blackbox)
        self.setInitX(init_x)
        self.setInitY(init_y)
        self._settings.set_all(categoricalVariables=categoricalVariables,
                               deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               initialDoeTechnique=initialDoeTechnique,
                               accelerator=accelerator, annealingCount=annealingCount, criterion=criterion,
                               exactFitRequired=exactFitRequired, internalValidation=internalValidation,
                               initialCount=initialCount, oneStepCount=oneStepCount, trainIterations=trainIterations)

    def getBlackbox(self):
        """
        Get the adaptive DoE blackbox.

        Returns
        -------
        blackbox: :class:`~openturns.NumericalMathFunction`
            Adaptive DoE blackbox.
        """
        return self.__blackbox

    def setBlackbox(self, blackbox):
        """
        Set the adaptive DoE blackbox.

        Parameters
        -------
        blackbox: :class:`~openturns.NumericalMathFunction`
            Adaptive DoE blackbox.
        """
        if not isinstance(blackbox, ot.NumericalMathFunction):
            raise TypeError("Wrong 'blackbox' type %s! Required: %s" %
                            (type(blackbox).__name__, ot.NumericalMathFunction))
        # Must be checked to prepare blackbox without fails (p7core gtdoe will do the same thing later).
        bounds_dimension = self.getBounds().getDimension()
        blackbox_dimension = blackbox.getInputDimension()
        if bounds_dimension != blackbox_dimension:
            raise ValueError("Inconsistent blackbox and bounds dimension")
        # Convert blackbox for p7 generator
        self.__p7_blackbox = _Blackbox(blackbox, self._settings.get('bounds'))
        self._settings.set('blackbox', self.__p7_blackbox)
        self.__blackbox = blackbox

    def getP7Blackbox(self):
        """
        Get the adaptive DoE blackbox in p7 format.

        Returns
        -------
        p7_blackbox: :class:`~da.p7core.blackbox.Blackbox`
            Adaptive DoE blackbox in p7 format.
        """
        return self.__p7_blackbox

    def getCount(self):
        """
        Get the number of points to generate.

        Returns
        -------
        count: int, long
            The number of points to generate.
        """
        # count is not for the blackbox-based adaptive DoE, use budget
        return self._settings.get('budget')

    def setCount(self, count):
        """
        Set the number of points to generate.

        Parameters
        ----------
        count: int, long
            The number of points to generate.
        """
        if not isinstance(count, (int, long)):
            raise TypeError("Wrong 'count' type %s! Required: int or long" % type(count).__name__)
        if count <= 0:
            raise ValueError("Argument 'count' must be > 0")
        # count is not for the blackbox-based adaptive DoE, use budget
        self._settings.set('budget', count)

    def getInitY(self):
        """
        Get the response part of initial sample for the adaptive DoE.

        Returns
        -------
        init_y: (array-like, 1D or 2D)
            Initial sample for the adaptive DoE, response part (function values).
        """
        return self._settings.get('init_y')

    def setInitY(self, init_y):
        """
        Set the response part of initial sample for the adaptive DoE.

        Parameters
        ----------
        init_y: (array-like, 1D or 2D)
            Initial sample for the adaptive DoE, response part (function values).
        """
        self._check_initial_sample(init_y)
        self._settings.set('init_y', init_y)

    def getInitialCount(self):
        """
        Get the size of initial sample.

        Returns
        -------
        initialCount: 0, or an integer in range [2*d_in+3, count] (except when initialDoeTechnique is "FullFactorial"),
                      where d_in is the design space dimensionality.
            In case initial training set was not provided by user, the tool generates a sample using the technique
            specified by the initialDoeTechnique option (Latin hypercube sampling by default). This sample is then
            evaluated with the blackbox, and the generated inputs and obtained blackbox outputs are used as an
            initial training set. initialCount sets the size of this sample. If left default, the size will be
            automatically set equal to 2*d_in+3, where d_in is the design space dimensionality (Default: 0).
        """
        return self._settings.get('initialCount')

    def setInitialCount(self, initialCount):
        """
        Set the size of initial sample.

        Parameters
        ----------
        initialCount: 0, or an integer in range [2*d_in+3, count] (except when initialDoeTechnique is "FullFactorial"),
                      where d_in is the design space dimensionality.
            In case initial training set was not provided by user, the tool generates a sample using the technique
            specified by the initialDoeTechnique option (Latin hypercube sampling by default). This sample is then
            evaluated with the blackbox, and the generated inputs and obtained blackbox outputs are used as an
            initial training set. initialCount sets the size of this sample. If left default, the size will be
            automatically set equal to 2*d_in+3, where d_in is the design space dimensionality (Default: 0).
        """
        self._settings.set('initialCount', initialCount)

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

    def getOneStepCount(self):
        """
        Get the number of points added to DoE on each iteration.

        Returns
        -------
        oneStepCount: integer in range [1, 2^31-2]
            Each adaptive DoE step may generate more than one point. This option sets the amount of points requested
            on each step. Note that it is not always possible to generate requested number of points; in such case
            maximum possible number of points is generated. This option takes effect in the blackbox-based adaptive
            DoE mode only. Sample-based adaptive DoE disregards this option (Default: 1).
        """
        return self._settings.get('oneStepCount')

    def setOneStepCount(self, oneStepCount):
        """
        Set the number of points added to DoE on each iteration.

        Parameters
        ----------
        oneStepCount: integer in range [1, 2^31-2]
            Each adaptive DoE step may generate more than one point. This option sets the amount of points requested
            on each step. Note that it is not always possible to generate requested number of points; in such case
            maximum possible number of points is generated. This option takes effect in the blackbox-based adaptive
            DoE mode only. Sample-based adaptive DoE disregards this option (Default: 1).
        """
        self._settings.set('oneStepCount', oneStepCount)

    def getTrainIterations(self):
        """
        Get the number of adaptive DoE iterations between rebuilds of approximation model.

        Returns
        -------
        trainIterations: integer in range [1, 2^31-2]
            Usually the approximation model used by adaptive DoE process is expected to only change slightly when a
            few points are added to the training set, so there is no need to rebuild the model at every step. This
            assumption, however, is not always true, especially when training set is not big enough to ensure that
            approximation has reasonable quality. This option sets the number of steps between rebuilds.
        """
        return self._settings.get('trainIterations')

    def setTrainIterations(self, trainIterations):
        """
        Set the number of adaptive DoE iterations between rebuilds of approximation model.

        Parameters
        ----------
        trainIterations: integer in range [1, 2^31-2]
            Usually the approximation model used by adaptive DoE process is expected to only change slightly when a
            few points are added to the training set, so there is no need to rebuild the model at every step. This
            assumption, however, is not always true, especially when training set is not big enough to ensure that
            approximation has reasonable quality. This option sets the number of steps between rebuilds.
        """
        self._settings.set('trainIterations', trainIterations)


class AdaptiveSample(_Adaptive):
    """
    Adaptive Sample-Based design of experiments.

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long
        The number of points to generate.
    init_x: (array-like, 1D or 2D)
        Initial sample for the adaptive DoE, input part (values of variables).
    categoricalVariables: a list in JSON format
        Specifies the indices of categorical variables and their categories for the GTDoE generator.
        Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
        index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
        and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    init_y: (array-like, 1D or 2D)
        Initial sample for the adaptive DoE, response part (function values).
    accelerator: integer in range [1, 5]
        Five-position switch to control the trade-off between speed and accuracy of approximations
        used by adaptive DoE (Default: 1).
    annealingCount: integer in range [1, 2^31-2]
        Each sample point added by adaptive DoE process is the result of optimization by selected
        adaptive DoE criterion. This optimization is iterative, and the more iterations it can make,
        the better the new point will probably be. This option adjusts the number of optimizer iterations.
        Note that it also directly affects the working time of adaptive DoE algorithm.
        Default (0) is an automatic estimate based on the design space dimensionality (d_in),
        which sets the number of iterations to min(500+100*d_in, 3000).
    criterion: "IntegratedMseGainMaxVar", "MaximumVariance", "Uniform", or "Auto"
        Control the behavior of adaptive generation algorithm. "IntegratedMseGainMaxVar": most accurate and
        time-consuming method. Estimates the error of approximation with new candidate point added to the sample
        and selects the point which minimizes the expected error. "MaximumVariance": samples points in the
        region with highest uncertainty, relying on model accuracy evaluation. Faster but less accurate.
        "Uniform": does not aim to increase model quality. Generates next sample point in such a way that
        overall sample is as uniform as possible. Note that in fact this is the only valid criterion for the
        sample-based adaptive DoE when only the input part of the initial sample is available. "Auto": defaults
        to "MaximumVariance" if both input and response parts of the initial sample are available, and
        to "Uniform" if only the input part is available.
    exactFitRequired: Boolean
        Require all approximations to fit the training data exactly. If this option is on (True),
        all approximations constructed in the adaptive DoE process fit the sample points exactly.
        If off (False) then no fitting condition is imposed (Default: False).
    internalValidation: Boolean
        Enable or disable internal validation for approximations used by adaptive DoE. Note that internal
        validation scores are computed for every model built, including intermediate models built between DoE
        iterations, so switching this on may significantly increase DoE generation time. Note that in the
        sample-based adaptive DoE mode, internal validation requires the response part of the initial sample.
        This is due to the fact that sample-based adaptive DoE with initial input part only is just random uniform
        generation which does not involve approximation models, so the option has no sense.
    """

    def __init__(self, bounds, count, init_x,
                 categoricalVariables=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 init_y=None, accelerator=None, annealingCount=None,
                 criterion=None, exactFitRequired=None, internalValidation=None):
        super(AdaptiveSample, self).__init__(bounds=bounds, count=count)
        init_y = [] if init_y is None else init_y
        self.setInitX(init_x)
        self.setInitY(init_y)
        self._settings.set_all(categoricalVariables=categoricalVariables,
                               deterministic=deterministic, seed=seed,
                               logLevel=logLevel, maxParallel=maxParallel,
                               accelerator=accelerator, annealingCount=annealingCount,
                               criterion=criterion, exactFitRequired=exactFitRequired,
                               internalValidation=internalValidation, technique='Adaptive')

    def getInitY(self):
        """
        Get the response part of initial sample for the adaptive DoE.

        Returns
        -------
        init_y: (array-like, 1D or 2D)
            Initial sample for the adaptive DoE, response part (function values).
        """
        return self._settings.get('init_y')

    def setInitY(self, init_y):
        """
        Set the response part of initial sample for the adaptive DoE.

        Parameters
        ----------
        init_y: (array-like, 1D or 2D)
            Initial sample for the adaptive DoE, response part (function values).
        """
        self._check_initial_sample(init_y)
        self._settings.set('init_y', init_y)


class AdaptiveLHS(LHS):
    """
    Adaptive LHS-Based design of experiments.

    Parameters
    ----------
    bounds: :class:`~openturns.Interval`
        Design space bounds.
    count: int, long
        The number of points to generate. It will be used to set the budget.
    init_x: (array-like, 1D or 2D)
        Initial sample for the adaptive DoE, input part (values of variables).
    categoricalVariables: a list in JSON format
        Specifies the indices of categorical variables and their categories for the GTDoE generator.
        Option value is a list in the following format: [idx, [ctg, ctg, ...], ...], where idx is a zero-based
        index of the variable in the list of blackbox variables or in the lists contained in the bounds tuple,
        and ctgs are category numbers (only int and float values are accepted as category numbers) (Default: []).
    deterministic: boolean
        Require generation to be reproducible (Default: False).
    seed: integer in range [1, 2^31 - 1]
        Fixed random seed (Default: 100). This option sets fixed seed value, which is used in all randomized
        algorithms if deterministic option is on. If deterministic is off, the seed value is ignored.
    logLevel: "Debug", "Info", "Warn", "Error" or "Fatal"
        Minimum log level (Default: "Info"). If this option is set,
        only messages with log level greater than or equal to the threshold are dumped into log.
    maxParallel: positive integer or 0
        Maximum number of parallel threads (Default: 0). GTDoE can run in parallel to speed up design generation.
        This option sets the maximum number of threads it is allowed to create. Default setting (0) uses the value
        given by the OMP_NUM_THREADS environment variable, which by default is equal to the number of virtual
        processors, including hyperthreading CPUs.
    useOptimized: boolean
        Use Optimized Latin Hypercube Sampling (Default: False).
    iterations: integer in range [2, 65535]
        Maximum number of optimization iterations in OLHS generation (Default: 300).
    """

    def __init__(self, bounds, count, init_x,
                 categoricalVariables=None,
                 deterministic=None, seed=None,
                 logLevel=None, maxParallel=None,
                 useOptimized=False, iterations=None):
        super(AdaptiveLHS, self).__init__(bounds=bounds, count=count,
                                          useOptimized=useOptimized, iterations=iterations,
                                          categoricalVariables=categoricalVariables,
                                          deterministic=deterministic, seed=seed,
                                          logLevel=logLevel, maxParallel=maxParallel)
        self.setInitX(init_x)

    def _check_initial_sample(self, sample):
        if not isinstance(sample, (list, tuple, np.ndarray, ot.NumericalSample, ot.NumericalPoint)):
            raise TypeError("Wrong initial sample type %s! Required: array-like" % type(sample).__name__)

    def getInitX(self):
        """
        Get the input part of initial sample for the adaptive DoE.

        Returns
        -------
        init_x: (array-like, 1D or 2D)
            Initial sample for the adaptive DoE, input part (values of variables).
        """
        return self._settings.get('init_x')

    def setInitX(self, init_x):
        """
        Set the input part of initial sample for the adaptive DoE.

        Parameters
        ----------
        init_x: (array-like, 1D or 2D)
            Initial sample for the adaptive DoE, input part (values of variables).
        """
        self._check_initial_sample(init_x)
        self._settings.set('init_x', init_x)
