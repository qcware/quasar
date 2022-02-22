Backends
--------
Choose a backend to run circuits over Forge using statevector simulation or
measurement-based computing. Refer to these Jupyter Notebook tutorials on Forge:

* ``GettingStarted/Examples/Circuits/start_here.ipynb``
* ``GettingStarted/Examples/Circuits/quasar_measurement.ipynb``

.. rubric:: Classes

* :py:class:`~quasar.backend.Backend`
* :py:class:`~quasar.backend.QuasarSimulatorBackend`
* :py:class:`~quasar.backend.QiskitHardwareBackend`
* :py:class:`~quasar.backend.QiskitSimulatorBackend`
* :py:class:`~quasar.backend.CirqSimulatorBackend`
* :py:class:`~quasar.measurement.CountHistogram`
* :py:class:`~quasar.measurement.ProbabilityHistogram`

.. rubric:: Class Backend

.. hlist::
   :columns: 3

   * :py:meth:`~quasar.backend.Backend.has_run_statevector`
   * :py:meth:`~quasar.backend.Backend.has_statevector_input`
   * :py:meth:`~quasar.backend.Backend.run_measurement`
   * :py:meth:`~quasar.backend.Backend.run_pauli_expectation_value`
   * :py:meth:`~quasar.backend.Backend.run_pauli_expectation_value_gradient`
   * :py:meth:`~quasar.backend.Backend.run_statevector`
   * :py:meth:`~quasar.backend.Backend.summary_str`

.. rubric:: Class QuasarSimulatorBackend

.. hlist::
   :columns: 3

   * :py:meth:`~quasar.quasar_backend.QuasarSimulatorBackend.has_run_statevector`
   * :py:meth:`~quasar.quasar_backend.QuasarSimulatorBackend.has_statevector_input`
   * :py:meth:`~quasar.quasar_backend.QuasarSimulatorBackend.run_measurement`
   * :py:meth:`~quasar.quasar_backend.QuasarSimulatorBackend.run_statevector`
   * :py:meth:`~quasar.quasar_backend.QuasarSimulatorBackend.summary_str`

.. rubric:: Class QiskitHardwareBackend

.. hlist::
   :columns: 3

   * :py:meth:`~quasar.qiskit_backend.QiskitHardwareBackend.has_run_statevector`
   * :py:meth:`~quasar.qiskit_backend.QiskitHardwareBackend.has_statevector_input`
   * :py:meth:`~quasar.qiskit_backend.QiskitHardwareBackend.run_measurement`
   * :py:meth:`~quasar.qiskit_backend.QiskitHardwareBackend.run_statevector`
   * :py:meth:`~quasar.qiskit_backend.QiskitHardwareBackend.summary_str`

.. rubric:: Class QiskitSimulatorBackend

.. hlist::
   :columns: 3

   * :py:meth:`~quasar.qiskit_backend.QiskitSimulatorBackend.has_run_statevector`
   * :py:meth:`~quasar.qiskit_backend.QiskitSimulatorBackend.has_statevector_input`
   * :py:meth:`~quasar.qiskit_backend.QiskitSimulatorBackend.run_measurement`
   * :py:meth:`~quasar.qiskit_backend.QiskitSimulatorBackend.run_statevector`
   * :py:meth:`~quasar.qiskit_backend.QiskitSimulatorBackend.summary_str`

.. rubric:: Class CirqSimulatorBackend

.. hlist::
   :columns: 3

   * :py:meth:`~quasar.cirq_backend.CirqSimulatorBackend.has_run_statevector`
   * :py:meth:`~quasar.cirq_backend.CirqSimulatorBackend.has_statevector_input`
   * :py:meth:`~quasar.cirq_backend.CirqSimulatorBackend.run_measurement`
   * :py:meth:`~quasar.cirq_backend.CirqSimulatorBackend.run_statevector`
   * :py:meth:`~quasar.cirq_backend.CirqSimulatorBackend.summary_str`

.. rubric:: Module Measurement

.. hlist::
   :columns: 2

   * :py:class:`~quasar.measurement.CountHistogram`
   * :py:class:`~quasar.measurement.ProbabilityHistogram`

Backend
^^^^^^^
   
.. automodule:: quasar.backend
   :members:


QuasarBackend
^^^^^^^^^^^^^
      
.. automodule:: quasar.quasar_backend
   :members:

QiskitBackend
^^^^^^^^^^^^^

.. automodule:: quasar.qiskit_backend
   :members:

CirqBackend
^^^^^^^^^^^

.. automodule:: quasar.cirq_backend
   :members:

Measurement
^^^^^^^^^^^

.. automodule:: quasar.measurement
   :members:
