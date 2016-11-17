The Python package p7ot for pSeven Core integration in OpenTURNS
===========================================================

The purpose of this package is to make [pSeven Core](https://www.datadvance.net/product/pseven-core/)
capabilities compatible with [OpenTURNS](http://openturns.org) (OT):

1.  To be able to use pSeven Core GTOpt in OpenTURNS for solving optimization problems
    that are defined and post-processed with the dedicated OT classes:
    OptimizationProblem, OptimizationSolver, and OptimizationResult.
2.  To be able to import surrogate models generated by pSeven Core GTApprox into
    OpenTURNS as an OT NumericalMathFunction.
3.  To be able to use pSeven Core GTDoE in OpenTURNS through the dedicated
    OT class Experiment.


REQUIREMENTS
------------

*   pSeven Core 6.9
*   OpenTURNS 1.7 (OpenTURNS 1.8rc2 for ExperimentImplementaion exchange)
*   numpy 1.6.1

p7ot has been tested with OpenTURNS 1.7 on Linux operating systems.      


QUICK START
-----------

Just write in Python

```python
>>> import openturns as ot
>>> import p7ot
```

#### Design of Experiment
Sequential design of experiment:
```python
>>> experiment = p7ot.Sequence(bounds=ot.Interval([0]*3, [10]*3), count=10, technique="SobolSeq")
>>> print experiment.generate()
0 : [ 0     0     0     ]
1 : [ 5     5     5     ]
2 : [ 7.5   2.5   2.5   ]
3 : [ 2.5   7.5   7.5   ]
4 : [ 3.75  3.75  6.25  ]
5 : [ 8.75  8.75  1.25  ]
6 : [ 6.25  1.25  8.75  ]
7 : [ 1.25  6.25  3.75  ]
8 : [ 1.875 3.125 9.375 ]
9 : [ 6.875 8.125 4.375 ]
```
Latin Hypercube Sampling (LHS) design of experiment:
```python
>>> experiment = p7ot.LHS(bounds=ot.Interval([0]*3, [10]*3), count=10, useOptimized=True)
>>> print experiment.generate()
0 : [ 1.5 7.5 6.5 ]
1 : [ 5.5 8.5 4.5 ]
2 : [ 4.5 4.5 8.5 ]
3 : [ 0.5 1.5 7.5 ]
4 : [ 8.5 6.5 2.5 ]
5 : [ 3.5 5.5 1.5 ]
6 : [ 6.5 0.5 3.5 ]
7 : [ 2.5 9.5 9.5 ]
8 : [ 7.5 3.5 0.5 ]
9 : [ 9.5 2.5 5.5 ]
```
Adaptive Blackbox-Based design of experiment:
```python
>>> blackbox = ot.NumericalMathFunction(['x1', 'x2', 'x3'], ['x1^2 + x2^2 + x3^2'])
>>> experiment = p7ot.AdaptiveBlackbox(blackbox=blackbox, bounds=ot.Interval([0]*3, [10]*3), count=10)
>>> print experiment.generate()
0 : [ 8.33333   5         0.555556  ]
1 : [ 0.555556  7.22222   1.66667   ]
2 : [ 6.11111   3.88889   5         ]
3 : [ 9.44444   8.33333   7.22222   ]
4 : [ 5         9.44444   6.11111   ]
5 : [ 1.66667   2.77778   9.44444   ]
6 : [ 7.22222   0.555556  8.33333   ]
7 : [ 3.88889   6.11111   3.88889   ]
8 : [ 2.77778   1.66667   2.77778   ]
9 : [ 0.0334393 9.99071   9.97424   ]
```

#### Approximation model
```python
>>> from da.p7core import gtapprox
>>> import numpy as np
>>> x = np.random.random((10, 2))
>>> y = [x1 * x2 for x1, x2 in x]
>>> p7_model = gtapprox.Builder().build(x,y)  # p7core model
>>> print p7_model.calc([0.2, 0.3])
[ 0.06139608]
>>> p7ot_function = p7ot.ModelFunction(p7_model)  # p7ot function
>>> print p7ot_function([0.2, 0.3])
[0.0613961]
>>> ot_function = ot.NumericalMathFunction(p7ot_function)  # OpenTURNS function
>>> print ot_function([0.2, 0.3])  
[0.0613961]
```

#### Optimization
```python
>>> rosenbrock_function = ot.NumericalMathFunction(['x1', 'x2', 'x3', 'x4'], ['100*(x4-x3^2)^2+(x3-1)^2+' +
...                                                                           '100*(x3-x2^2)^2+(x2-1)^2+' +
...                                                                           '100*(x2-x1^2)^2+(x1-1)^2'])
>>> bounds = ot.Interval([-500]*4, [500]*4)
>>> problem = ot.OptimizationProblem()
>>> problem.setObjective(rosenbrock_function)
>>> problem.setBounds(bounds)
>>> solver = p7ot.GTOpt(problem)  # p7core optimization algorithm
>>> solver.setStartingPoint(bounds.getUpperBound())
>>> solver.run()
>>> result = solver.getResult()
>>> print result.getIterationNumber()
1096
>>> print result.getOptimalPoint()
[0.999933,0.999867,0.999735,0.999468]
>>> print result.getOptimalValue()
[9.29851e-08]
```

DOCUMENTATION
-------------

For any additional information about algorithms used and options description see
[pSeven Core Documentation](https://www.datadvance.net/product/macros/manual/6.9/index.html)

LICENSE
-------

p7ot is distributed under the Lesser General Public License.  
Please see the LICENSE.GPL, LICENSE.LGPL files for details of the license.   
