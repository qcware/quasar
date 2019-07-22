Circuit
=======

A :py:class:`~quasar.Circuit` is effectively a list of gates and instructions which
can be created using the function documented in this section::

    >>> import quasar
    >>>
    >>> circuit = quasar.Circuit(N=2)
    >>> circuit = circuit.X(0).CX(0,1)

.. autoclass:: quasar.Circuit
.. autofunction:: quasar.Circuit.__init__

.. rubric:: Properties
.. autosummary::
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

.. rubric:: Circuits/Gates
.. autosummary::
    ~quasar.Circuit.is_equivalent
    ~quasar.Circuit.sort_gates
    ~quasar.Circuit.add_gate
    ~quasar.Circuit.gate   

.. rubric:: Simulation/Measurement
.. autosummary::
    ~quasar.Circuit.simulate
    ~quasar.Circuit.measure 
    
.. rubric:: Copy/Subsets/Concatenation
.. autosummary::
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

.. rubric:: Parameters
.. autosummary::
    ~quasar.Circuit.set_param_values     
    ~quasar.Circuit.set_params
    
.. rubric:: Gate Addition
.. autosummary::
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

    
    
    
.. autofunction:: quasar.Circuit.is_equivalent

    
    
        