.. currentmodule:: quasar

Matrix
======

.. autoclass:: Matrix

Common Matrices
---------------
.. autosummary::
    :toctree: _autosummary
    
    ~quasar.Matrix.I
    ~quasar.Matrix.X
    ~quasar.Matrix.Y
    ~quasar.Matrix.Z
    ~quasar.Matrix.S
    ~quasar.Matrix.T
    ~quasar.Matrix.H
    ~quasar.Matrix.Rx2
    ~quasar.Matrix.Rx2T
    ~quasar.Matrix.II
    ~quasar.Matrix.IX
    ~quasar.Matrix.IY
    ~quasar.Matrix.IZ
    ~quasar.Matrix.XI
    ~quasar.Matrix.XX
    ~quasar.Matrix.XY
    ~quasar.Matrix.XZ
    ~quasar.Matrix.YI
    ~quasar.Matrix.YX
    ~quasar.Matrix.YY
    ~quasar.Matrix.YZ
    ~quasar.Matrix.ZI
    ~quasar.Matrix.ZX
    ~quasar.Matrix.ZY
    ~quasar.Matrix.ZZ
    ~quasar.Matrix.CX
    ~quasar.Matrix.CY
    ~quasar.Matrix.CZ
    ~quasar.Matrix.CS
    ~quasar.Matrix.SWAP
    ~quasar.Matrix.CSWAP
    ~quasar.Matrix.CCX
    ~quasar.Matrix.Rx
    ~quasar.Matrix.Ry
    ~quasar.Matrix.Rz
    ~quasar.Matrix.u1
    ~quasar.Matrix.u2
    ~quasar.Matrix.u3
    ~quasar.Matrix.R_ion
    ~quasar.Matrix.Rz_ion
    ~quasar.Matrix.XX_ion
    
Gate
====

.. autoclass:: Gate
.. autofunction:: quasar.Gate.__init__

Properties
----------
.. autosummary::
    :toctree: _autosummary
    
    ~quasar.Gate.U

ControlledGate
==============

.. autoclass:: ControlledGate
.. autofunction:: quasar.ControlledGate.__init__

Properties
----------
.. autosummary::
    :toctree: _autosummary
    
    ~quasar.ControlledGate.U


Methods
-------

.. autosummary::
    :toctree: _autosummary

    ~quasar.Gate.copy
    ~quasar.Gate.same_unitary
    ~quasar.Gate.set_param
    ~quasar.Gate.set_params

Static Methods
--------------
.. autosummary::
    :toctree: _autosummary
    
    ~quasar.Gate.I
    ~quasar.Gate.X
    ~quasar.Gate.Y
    ~quasar.Gate.Z
    ~quasar.Gate.S
    ~quasar.Gate.T
    ~quasar.Gate.H
    ~quasar.Gate.Rx2
    ~quasar.Gate.Rx2T

Circuit
=======

.. autoclass:: Circuit
.. autofunction:: quasar.Circuit.__init__

Properties
----------
.. autosummary::
    :toctree: _autosummary
    
    ~quasar.Circuit.ntime
    ~quasar.Circuit.ngate
    ~quasar.Circuit.ngate1
    ~quasar.Circuit.ngate2
    ~quasar.Circuit.params
    ~quasar.Circuit.set_params
    ~quasar.Circuit.param_str
    ~quasar.Circuit.nparam
    ~quasar.Circuit.param_keys
    ~quasar.Circuit.param_values
    ~quasar.Circuit.param_str
    
Circuits/Gates
--------------
.. autosummary::
    :toctree: _autosummary
 
    ~quasar.Circuit.is_equivalent
    ~quasar.Circuit.sort_gates
    ~quasar.Circuit.add_gate
    ~quasar.Circuit.gate   
    
Simulation/Measurement
----------------------
.. autosummary::
    :toctree: _autosummary
    
    ~quasar.Circuit.simulate
    ~quasar.Circuit.measure 
    
Copy/Subsets/Concatenation
--------------------------
.. autosummary::
    :toctree: _autosummary
 
    ~quasar.Circuit.copy
    ~quasar.Circuit.subset
    ~quasar.Circuit.concatenate
    ~quasar.Circuit.deadjoin
    ~quasar.Circuit.adjoin
    ~quasar.Circuit.reversed
    ~quasar.Circuit.nonredundant
    ~quasar.Circuit.compressed
    ~quasar.Circuit.subcircuit
    ~quasar.Circuit.add_circuit
    
Parameters
----------
.. autosummary::
    :toctree: _autosummary
 
    ~quasar.Circuit.set_param_values     
    ~quasar.Circuit.set_params
    
Gate Addition
-------------
.. autosummary::
    :toctree: _autosummary
 
    ~quasar.Circuit.I
    ~quasar.Circuit.X
    ~quasar.Circuit.Y
    ~quasar.Circuit.Z
    ~quasar.Circuit.H    
    ~quasar.Circuit.S
    ~quasar.Circuit.T
    ~quasar.Circuit.CX
    ~quasar.Circuit.CY      
    ~quasar.Circuit.CZ
    ~quasar.Circuit.CS
    ~quasar.Circuit.SWAP      
    ~quasar.Circuit.CCX
    ~quasar.Circuit.CSWAP
    ~quasar.Circuit.Rx
    ~quasar.Circuit.Ry
    ~quasar.Circuit.Rz
    
    
        
