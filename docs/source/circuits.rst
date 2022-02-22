Circuits and Gates
-------------------
Construct and manipulate circuits. Refer to these Jupyter Notebook tutorials on Forge:

* ``GettingStarted/Examples/Circuits/quasar_circuit_composition.ipynb``
* ``GettingStarted/Examples/Circuits/quasar_gate_library.ipynb``
* ``GettingStarted/Examples/Circuits/quasar_parameters.ipynb``

.. rubric:: Classes

* :py:class:`~quasar.circuit.Circuit`
* :py:class:`~quasar.circuit.CompositeGate`
* :py:class:`~quasar.circuit.ControlledGate`
* :py:class:`~quasar.circuit.Gate`
* :py:class:`~quasar.circuit.Matrix` 

Class Circuit
^^^^^^^^^^^^^

.. rubric:: Circuit Attributes

.. hlist::
   :columns: 5

   * :py:meth:`~quasar.circuit.Circuit.is_composite`
   * :py:meth:`~quasar.circuit.Circuit.is_controlled`
   * :py:meth:`~quasar.circuit.Circuit.max_gate_nqubit`
   * :py:meth:`~quasar.circuit.Circuit.max_gate_ntime`
   * :py:meth:`~quasar.circuit.Circuit.max_qubit`
   * :py:meth:`~quasar.circuit.Circuit.max_time`
   * :py:meth:`~quasar.circuit.Circuit.min_qubit`
   * :py:meth:`~quasar.circuit.Circuit.min_time`
   * :py:meth:`~quasar.circuit.Circuit.ngate`
   * :py:meth:`~quasar.circuit.Circuit.ngate1`
   * :py:meth:`~quasar.circuit.Circuit.ngate2`
   * :py:meth:`~quasar.circuit.Circuit.ngate3`
   * :py:meth:`~quasar.circuit.Circuit.ngate4`
   * :py:meth:`~quasar.circuit.Circuit.ngate_nqubit`
   * :py:meth:`~quasar.circuit.Circuit.nqubit`
   * :py:meth:`~quasar.circuit.Circuit.nqubit_sparse`
   * :py:meth:`~quasar.circuit.Circuit.ntime`
   * :py:meth:`~quasar.circuit.Circuit.ntime_sparse`

.. rubric:: Circuit Composition

.. hlist::
   :columns: 5

   * :py:meth:`~quasar.circuit.Circuit.add_gate`
   * :py:meth:`~quasar.circuit.Circuit.add_gates`
   * :py:meth:`~quasar.circuit.Circuit.adjoint`
   * :py:meth:`~quasar.circuit.Circuit.center`
   * :py:meth:`~quasar.circuit.Circuit.copy`
   * :py:meth:`~quasar.circuit.Circuit.join_in_qubits`
   * :py:meth:`~quasar.circuit.Circuit.join_in_time`
   * :py:meth:`~quasar.circuit.Circuit.remove_gate`
   * :py:meth:`~quasar.circuit.Circuit.replace_gate`
   * :py:meth:`~quasar.circuit.Circuit.reverse`
   * :py:meth:`~quasar.circuit.Circuit.slice`
   * :py:meth:`~quasar.circuit.Circuit.sparse`

.. rubric:: Circuit Parameter Manipulation

.. hlist::
   :columns: 4

   * :py:meth:`~quasar.circuit.Circuit.nparameter`
   * :py:meth:`~quasar.circuit.Circuit.parameters`
   * :py:meth:`~quasar.circuit.Circuit.parameter_indices`
   * :py:meth:`~quasar.circuit.Circuit.parameter_keys`
   * :py:meth:`~quasar.circuit.Circuit.parameter_str`
   * :py:meth:`~quasar.circuit.Circuit.parameter_values`
   * :py:meth:`~quasar.circuit.Circuit.set_parameter`
   * :py:meth:`~quasar.circuit.Circuit.set_parameters`
   * :py:meth:`~quasar.circuit.Circuit.set_parameter_values`

.. rubric:: Gates

.. hlist::
   :columns: 8 

   * :py:meth:`~quasar.circuit.Circuit.CCX`
   * :py:meth:`~quasar.circuit.Circuit.CF`
   * :py:meth:`~quasar.circuit.Circuit.CS`
   * :py:meth:`~quasar.circuit.Circuit.CST`
   * :py:meth:`~quasar.circuit.Circuit.CSWAP`
   * :py:meth:`~quasar.circuit.Circuit.CX`
   * :py:meth:`~quasar.circuit.Circuit.CY`
   * :py:meth:`~quasar.circuit.Circuit.CZ`
   * :py:meth:`~quasar.circuit.Circuit.H`
   * :py:meth:`~quasar.circuit.Circuit.I`
   * :py:meth:`~quasar.circuit.Circuit.R_ion`
   * :py:meth:`~quasar.circuit.Circuit.Rx`
   * :py:meth:`~quasar.circuit.Circuit.Rx2`
   * :py:meth:`~quasar.circuit.Circuit.Rx2T`
   * :py:meth:`~quasar.circuit.Circuit.Rx_ion`
   * :py:meth:`~quasar.circuit.Circuit.Ry`
   * :py:meth:`~quasar.circuit.Circuit.Ry_ion`
   * :py:meth:`~quasar.circuit.Circuit.Rz`
   * :py:meth:`~quasar.circuit.Circuit.Rz_ion`
   * :py:meth:`~quasar.circuit.Circuit.S`
   * :py:meth:`~quasar.circuit.Circuit.SO4`
   * :py:meth:`~quasar.circuit.Circuit.S042`
   * :py:meth:`~quasar.circuit.Circuit.ST`
   * :py:meth:`~quasar.circuit.Circuit.SWAP`
   * :py:meth:`~quasar.circuit.Circuit.T`
   * :py:meth:`~quasar.circuit.Circuit.TT`
   * :py:meth:`~quasar.circuit.Circuit.U1`
   * :py:meth:`~quasar.circuit.Circuit.U2`
   * :py:meth:`~quasar.circuit.Circuit.u1`
   * :py:meth:`~quasar.circuit.Circuit.u2`
   * :py:meth:`~quasar.circuit.Circuit.u3`
   * :py:meth:`~quasar.circuit.Circuit.X`
   * :py:meth:`~quasar.circuit.Circuit.XX_ion`
   * :py:meth:`~quasar.circuit.Circuit.Y`
   * :py:meth:`~quasar.circuit.Circuit.Z`

.. rubric:: Other

* :py:meth:`~quasar.circuit.Circuit.test_equivalence`

Class CompositeGate
^^^^^^^^^^^^^^^^^^^

.. hlist:: 
    :columns: 4

    * :py:meth:`~quasar.circuit.CompositeGate.adjoint`
    * :py:meth:`~quasar.circuit.CompositeGate.apply_to_statevector`
    * :py:meth:`~quasar.circuit.CompositeGate.copy`
    * :py:meth:`~quasar.circuit.CompositeGate.is_composite`
    * :py:meth:`~quasar.circuit.CompositeGate.is_controlled`
    * :py:meth:`~quasar.circuit.CompositeGate.ntime`
    * :py:meth:`~quasar.circuit.CompositeGate.set_parameter`
    * :py:meth:`~quasar.circuit.CompositeGate.set_parameters`

Class ControlledGate
^^^^^^^^^^^^^^^^^^^^

.. hlist::
    :columns: 4

    * :py:meth:`~quasar.circuit.ControlledGate.adjoint`
    * :py:meth:`~quasar.circuit.ControlledGate.copy`
    * :py:meth:`~quasar.circuit.ControlledGate.is_composite`
    * :py:meth:`~quasar.circuit.ControlledGate.is_controlled`
    * :py:meth:`~quasar.circuit.ControlledGate.ntime`
    * :py:meth:`~quasar.circuit.ControlledGate.set_parameter`
    * :py:meth:`~quasar.circuit.ControlledGate.set_parameters`

Class Gate
^^^^^^^^^^

.. rubric:: Explicit 1-Body Gates

.. hlist::
   :columns: 11

   * :py:attr:`~quasar.circuit.Gate.I`
   * :py:attr:`~quasar.circuit.Gate.H`
   * :py:attr:`~quasar.circuit.Gate.Rx2`
   * :py:attr:`~quasar.circuit.Gate.Rx2T`
   * :py:attr:`~quasar.circuit.Gate.S`
   * :py:attr:`~quasar.circuit.Gate.ST`
   * :py:attr:`~quasar.circuit.Gate.T`
   * :py:attr:`~quasar.circuit.Gate.TT`
   * :py:attr:`~quasar.circuit.Gate.X`
   * :py:attr:`~quasar.circuit.Gate.Y`
   * :py:attr:`~quasar.circuit.Gate.Z`

.. rubric:: Explicit 2-Body Gates

.. hlist::
   :columns: 6

   * :py:attr:`~quasar.circuit.Gate.CS`
   * :py:attr:`~quasar.circuit.Gate.CST`
   * :py:attr:`~quasar.circuit.Gate.CX`
   * :py:attr:`~quasar.circuit.Gate.CY`
   * :py:attr:`~quasar.circuit.Gate.CZ`
   * :py:attr:`~quasar.circuit.Gate.SWAP`

.. rubric:: Explicit 3-Body Gates

.. hlist::
   :columns: 2

   * :py:attr:`~quasar.circuit.Gate.CCX`
   * :py:attr:`~quasar.circuit.Gate.CSWAP`

.. rubric:: Parameterized 1-Body Gates

.. hlist::
   :columns: 8

   * :py:meth:`~quasar.circuit.Gate.RBS`
   * :py:meth:`~quasar.circuit.Gate.iRBS`
   * :py:meth:`~quasar.circuit.Gate.Rx`
   * :py:meth:`~quasar.circuit.Gate.Ry`
   * :py:meth:`~quasar.circuit.Gate.Rz`
   * :py:meth:`~quasar.circuit.Gate.u1`
   * :py:meth:`~quasar.circuit.Gate.u2`
   * :py:meth:`~quasar.circuit.Gate.u3`

.. rubric:: Parameterized 2-Body Gates

.. hlist::
   :columns: 3

   * :py:meth:`~quasar.circuit.Gate.CF`
   * :py:meth:`~quasar.circuit.Gate.SO4`
   * :py:meth:`~quasar.circuit.Gate.SO42`

.. rubric:: Special Explicit Gates

.. hlist::
   :columns: 2

   * :py:meth:`~quasar.circuit.Gate.U1`
   * :py:meth:`~quasar.circuit.Gate.U2`

.. rubric:: Gate Attributes

.. hlist::
   :columns: 5

   * :py:meth:`~quasar.circuit.Gate.is_composite`
   * :py:meth:`~quasar.circuit.Gate.is_controlled`
   * :py:meth:`~quasar.circuit.Gate.nparameter`
   * :py:meth:`~quasar.circuit.Gate.ntime`
   * :py:meth:`~quasar.circuit.Gate.operator`

.. rubric:: Other

.. hlist::
   :columns: 4

   * :py:meth:`~quasar.circuit.Gate.adjoint`
   * :py:meth:`~quasar.circuit.Gate.apply_to_statevector`
   * :py:meth:`~quasar.circuit.Gate.copy`
   * :py:meth:`~quasar.circuit.Gate.set_parameter`
   * :py:meth:`~quasar.circuit.Gate.set_parameters`
   * :py:meth:`~quasar.circuit.Gate.test_operator_equivalence`

Class Matrix
^^^^^^^^^^^^

.. hlist::
    :columns: 8

    * :py:meth:`~quasar.circuit.Matrix.CCX`
    * :py:meth:`~quasar.circuit.Matrix.CS`
    * :py:meth:`~quasar.circuit.Matrix.CST`
    * :py:meth:`~quasar.circuit.Matrix.CSWAP`
    * :py:meth:`~quasar.circuit.Matrix.CX`
    * :py:meth:`~quasar.circuit.Matrix.CY`
    * :py:meth:`~quasar.circuit.Matrix.CZ`
    * :py:meth:`~quasar.circuit.Matrix.H`
    * :py:meth:`~quasar.circuit.Matrix.I`
    * :py:meth:`~quasar.circuit.Matrix.II`
    * :py:meth:`~quasar.circuit.Matrix.IX`
    * :py:meth:`~quasar.circuit.Matrix.IY`
    * :py:meth:`~quasar.circuit.Matrix.IZ`
    * :py:meth:`~quasar.circuit.Matrix.Rx`
    * :py:meth:`~quasar.circuit.Matrix.Rx2`
    * :py:meth:`~quasar.circuit.Matrix.Rx2T`
    * :py:meth:`~quasar.circuit.Matrix.Ry`
    * :py:meth:`~quasar.circuit.Matrix.Rz`
    * :py:meth:`~quasar.circuit.Matrix.S`
    * :py:meth:`~quasar.circuit.Matrix.ST`
    * :py:meth:`~quasar.circuit.Matrix.SWAP`
    * :py:meth:`~quasar.circuit.Matrix.T`
    * :py:meth:`~quasar.circuit.Matrix.TT`
    * :py:meth:`~quasar.circuit.Matrix.X`
    * :py:meth:`~quasar.circuit.Matrix.XI`
    * :py:meth:`~quasar.circuit.Matrix.XX`
    * :py:meth:`~quasar.circuit.Matrix.XY`
    * :py:meth:`~quasar.circuit.Matrix.XZ`
    * :py:meth:`~quasar.circuit.Matrix.Y`
    * :py:meth:`~quasar.circuit.Matrix.YI`
    * :py:meth:`~quasar.circuit.Matrix.YX`
    * :py:meth:`~quasar.circuit.Matrix.YY`
    * :py:meth:`~quasar.circuit.Matrix.YZ`
    * :py:meth:`~quasar.circuit.Matrix.Z`
    * :py:meth:`~quasar.circuit.Matrix.ZI`
    * :py:meth:`~quasar.circuit.Matrix.ZX`
    * :py:meth:`~quasar.circuit.Matrix.ZY`
    * :py:meth:`~quasar.circuit.Matrix.ZZ`

.. module:: quasar.circuit

.. autoclass:: Circuit
   :members:

.. autoclass:: CompositeGate
   :members:

.. autoclass:: ControlledGate
   :members:

.. autoclass:: Gate
   :members:

   .. autoattribute:: I
      :annotation:

      Static attribute representing the I (identity) gate.

   .. autoattribute:: X
      :annotation:

      Static attribute representing the X (NOT) gate.

   .. autoattribute:: Y
      :annotation:

      Static attribute representing the Y gate.

   .. autoattribute:: Z
      :annotation:

      Static attribute representing the Z gate.

   .. autoattribute:: H 
      :annotation:

      Static attribute representing the H gate.

   .. autoattribute:: S
      :annotation:

      Static attribute representing the S gate.

   .. autoattribute:: ST
      :annotation:

      Static attribute representing the S^+ gate.

   .. autoattribute:: T
      :annotation:

      Static attribute representing the T gate.

   .. autoattribute:: TT
      :annotation:
   
      Static attribute representing the T^+ gate.

   .. autoattribute:: Rx2
      :annotation:

      Static attribute representing the Rx2 gate.

   .. autoattribute:: Rx2T
      :annotation:

      Static attribute representing the Rx2T gate.

   .. autoattribute:: CX 
      :annotation:

      Static attribute representing the CX (CNOT) gate.

   .. autoattribute:: CY
      :annotation:

      Static attribute representing the CY gate.

   .. autoattribute:: CZ
      :annotation:

      Static attribute representing the CZ gate.

   .. autoattribute:: CS 
      :annotation:

      Static attribute representing the CS gate.

   .. autoattribute:: CST
      :annotation:

      Static attribute representing the CS^T+ gate.

   .. autoattribute:: SWAP
      :annotation:

      Static attribute representing the SWAP gate.

   .. autoattribute:: CCX
      :annotation:

      Static attribute representing the CCX (Toffoli) gate.

   .. autoattribute:: CSWAP
      :annotation:

      Static attribute representing the CSWAP (Toffoli) gate.

.. autoclass:: Matrix
   :members: