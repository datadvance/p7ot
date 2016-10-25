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
from da.p7core import gtopt
from .problem import _ProblemGeneric


class GTOpt(ot.OptimizationSolverImplementation):
    """
    Generic Tool for Optimization (GTOpt) module from pSeven Core

    Available constructors:
        GTOpt(*problem, options=None, input_hints=None, objective_hints=None, equality_hints=None,
               inequality_hints=None, sample_x=None, sample_f=None, sample_c=None*)

    Parameters
    ----------
    problem : :class:`~openturns.OptimizationProblem
        Optimization problem.
    options : dictionary
        Solver options.
    input_hints : list of dictionaries or None
        Additional properties of defined design variables.
    objective_hints : list of dictionaries or None
        Additional properties of defined objective functions.
    equality_hints : list of dictionaries or None
        Additional properties of defined equality constraints.
    inequality_hints : list of dictionaries or None
        Additional properties of defined inequality constraints.
    sample_x : array-like, 1D or 2D
        Optional initial sample containing values of variables.
    sample_f : array-like, 1D or 2D
        Optional initial sample of objective function values, requires sample_x.
    sample_c : array-like, 1D or 2D
        Optional initial sample of constraint function values, requires sample_x.
    """
    def __init__(self, problem, options=None,
                 input_hints=None, objective_hints=None, equality_hints=None, inequality_hints=None,
                 sample_x=None, sample_f=None, sample_c=None):
        super(GTOpt, self).__init__(problem)
        self.__options = gtopt.Solver().options
        if options:
            self.__options.set(options)
        self.__sample_x = sample_x
        self.__sample_f = sample_f
        self.__sample_c = sample_c
        self.__use_objectives_gradient = False
        self.__use_constraints_gradient = False
        # Dimentions
        self.__input_dim = problem.getDimension()
        self.__objectives_dim = problem.getObjective().getOutputDimension()
        self.__equality_dim = problem.getEqualityConstraint().getOutputDimension()
        self.__inequality_dim = problem.getInequalityConstraint().getOutputDimension()
        # GTOpt hints
        self.__input_hints = self.__add_hints(dimension=self.__input_dim, hints_list=input_hints)
        self.__objective_hints = self.__add_hints(dimension=self.__objectives_dim, hints_list=objective_hints)
        self.__equality_hints = self.__add_hints(dimension=self.__equality_dim, hints_list=equality_hints)
        self.__inequality_hints = self.__add_hints(dimension=self.__inequality_dim, hints_list=inequality_hints)

    def getP7Result(self):
        """
        Accessor to p7core optimization result.

        Returns
        -------
        p7result : :class:`~da.p7core.gtopt.Result`
            Result object.
        """
        return self.__p7_result

    def getClassName(self):
        """
        Accessor to the object's name.

        Returns
        -------
        class_name : str
            The object class name (`object.__class__.__name__`).
        """
        return self.__class__.__name__

    def run(self):
        """Launch the optimization."""
        p7_problem = _ProblemGeneric(problem=self.getProblem(), starting_point=self.getStartingPoint(),
                                     use_objectives_gradient=self.__use_objectives_gradient,
                                     use_constraints_gradient=self.__use_constraints_gradient,
                                     input_hints=self.__input_hints, objective_hints=self.__objective_hints,
                                     equality_hints=self.__equality_hints, inequality_hints=self.__inequality_hints)
        p7_problem.set_history(memory=True)
        p7_result = gtopt.Solver().solve(p7_problem, options=self.__options.get(),
                                         sample_x=self.__sample_x, sample_f=self.__sample_f, sample_c=self.__sample_c)
        # p7core.gtopt doesn't support maximization problem
        if not self.getProblem().isMinimization():
            p7_result.optimal.f[0] = [-1 * i if i else None for i in p7_result.optimal.f[0]]
        # Convert p7 result object to openturns result
        ot_result = self.__get_ot_result(p7_result, p7_problem)
        self.setResult(ot_result)

    def getMaximumIterationNumber(self):
        """
        Accessor to maximum allowed number of iterations.

        Returns
        -------
        N : int
            Maximum allowed number of iterations.
        """
        # Default value is 0
        self.__options.get('GTOpt/MaximumIterations')

    def setMaximumIterationNumber(self, value):
        """
        Accessor to maximum allowed number of iterations.

        Parameters
        ----------
        N : int
            Maximum allowed number of iterations.
        """
        self.__options.set('GTOpt/MaximumIterations', value)

    def getVerbose(self):
        """
        Accessor to the verbosity flag.

        Returns
        ----------
        verbose : bool
            Verbosity flag state.
        """
        # Default value is False
        self.__options.get('GTOpt/VerboseOutput')

    def setVerbose(self, value):
        """
        Accessor to the verbosity flag.

        Parameters
        ----------
        verbose : bool
            Verbosity flag state.
        """
        self.__options.set('GTOpt/VerboseOutput', value)

    def enableObjectivesGradient(self):
        """Enable the use of analytical gradients of objective functions."""
        self.__use_objectives_gradient = True

    def disableObjectivesGradient(self):
        """Disable the use of analytical gradients of objective functions."""
        self.__use_objectives_gradient = False

    def isObjectivesGradientEnabled(self):
        """
        Test whether the analytical gradients of objective functions are enabled or not.

        Returns
        -------
        useObjectivesGradient : bool
            Flag telling whether the analytical gradients of objective functions are enabled.
            It is disabled by default.
        """
        return self.__use_objectives_gradient

    def enableConstraintsGradient(self):
        """Enable the use of analytical gradients of constraint functions."""
        self.__use_constraints_gradient = True

    def disableConstraintsGradient(self):
        """Disable the use of analytical gradients of constraint functions."""
        self.__use_constraints_gradient = False

    def isConstraintsGradientEnabled(self):
        """
        Test whether the analytical gradients of constraint functions are enabled or not.

        Returns
        -------
        useConstraintGradient : bool
            Flag telling whether the analytical gradients of constraint functions are enabled.
            It is disabled by default.
        """
        return self.__use_constraints_gradient

    def setInputHints(self, hints, indices=None):
        """
        Set additional properties of defined design variables.

        Parameters
        ----------
        hints : dictionary
            Hints (dictionary keys) and their values.
        indices : list of integers or None (all of them)
            Set of indices for which the hints will be applied.
        """
        self.__input_hints = self.__add_hints(hints_list=self.__input_hints, dimension=self.__input_dim,
                                              hints=hints, indices=indices)

    def setObjectiveHints(self, hints, indices=None):
        """
        Set additional properties of defined objective functions.

        Parameters
        ----------
        hints : dictionary
            Hints (dictionary keys) and their values.
        indices : list of integers or None (all of them)
            Set of indices for which the hints will be applied.
        """
        self.__objective_hints = self.__add_hints(hints_list=self.__objective_hints, dimension=self.__objectives_dim,
                                                  hints=hints, indices=indices)

    def setEqualityConstraintHints(self, hints, indices=None):
        """
        Set additional properties of defined equality constraints.

        Parameters
        ----------
        hints : dictionary
            Hints (dictionary keys) and their values.
        indices : list of integers or None (all of them)
            Set of indices for which the hints will be applied.
        """
        self.__equality_hints = self.__add_hints(hints_list=self.__equality_hints, dimension=self.__equality_dim,
                                                 hints=hints, indices=indices)

    def setInequalityConstraintHints(self, hints, indices=None):
        """
        Set additional properties of defined inequality constraints.

        Parameters
        ----------
        hints : dictionary
            Hints (dictionary keys) and their values.
        indices : list of integers or None (all of them)
            Set of indices for which the hints will be applied.
        """
        self.__inequality_hints = self.__add_hints(hints_list=self.__inequality_hints, dimension=self.__inequality_dim,
                                                   hints=hints, indices=indices)

    # Check arguments and create valid list of hints that are to be used to prepare p7 problem.
    # hints_list will be used as a basis
    # hints will be applied to the elements with the corresponding indices of the hints_list.
    def __add_hints(self, hints_list, dimension, hints=None, indices=None):
        # Check hints_list if specified
        if hints_list is None:
            # Default result
            hints_list = [{}] * dimension
        else:
            # hints_list should be a list
            if type(hints_list) is not list:
                raise TypeError("Wrong type of hints list. Expected: list of dictionaries, got: %s" % type(hints_list))
            # Check the length of the hints_list
            if len(hints_list) != dimension:
                raise ValueError("Wrong length of hints list, Expected: %s, got: %s" % (dimension, len(hints_list)))
            # Elements of the hints_list should be a dictionary
            invalid_hints = [type(item) for item in hints_list if type(item) is not dict]
            if invalid_hints:
                raise TypeError("Wrong type of hints. Expected: dictionary, got: %s" % invalid_hints)
        # Check hints if specified, otherwise return hints_list without modifying
        if hints is None and indices is None:
            return hints_list
        elif type(hints) is not dict:
            # hints should be a dictionary
            raise TypeError("Wrong type of hints. Expected: dictionary, got: %s" % type(hints))
        # Check indicies if specified
        if indices is None:
            # hints will be set to each element of the hints_list
            indices = range(dimension)
        else:
            # indices should be a list of integers or a single integer
            if type(indices) is int:
                # In case of getting a single index instead of an indices list
                indices = [indices]
            elif type(indices) is list:
                # Remove dublicates from the indicies list
                indices = list(set(indices))
                # Indices should be in valid range [0, dimension-1]
                invalid_indices = [item for item in indices if item not in range(dimension)]
                if invalid_indices:
                    raise ValueError("The indices must be in range 0...%s, got %s" % (dimension-1, invalid_indices))
            else:
                raise TypeError("Wrong type of indices. Expected: list or integer, got: %s" % (type(indices)))
        # Modify the resulting list of hints
        for i in indices:
            hints_list[i] = hints
        return hints_list

    def __get_ot_result(self, p7_result, p7_problem):
        iteration_number = len(p7_problem.history)
        optimal_point = ot.NumericalPoint(p7_result.optimal.x[0])
        # In case of optimization problem with level function g(x)=v:
        # 1. level function g(x)=v acts as equality constraint, min||x|| - as objective function
        # 2. the value of level function should be set as optimal
        if self.getProblem().hasLevelFunction():
            optimal_value = self.getProblem().getLevelFunction()(optimal_point)
        else:
            optimal_value = ot.NumericalPoint(p7_result.optimal.f[0])
        # Absolute, constraint, residual and relative errors cal p7 builder
        # for any available information about the solving process see getP7Result() and da.p7core.gtopt.Result API
        return ot.OptimizationResult(optimal_point, optimal_value, iteration_number, -1, -1, -1, -1)
