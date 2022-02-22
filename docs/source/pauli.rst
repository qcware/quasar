Pauli Operators
---------------

Represent and manipulate Pauli operators. Evaluate Pauli expectation values
using ideal statevector simulation. Refer to these Jupyter Notebook tutorials on Forge:

* ``GettingStarted/Examples/Circuits/quasar_derivatives.ipynb``
* ``GettingStarted/Examples/Circuits/quasar_parameters.ipynb``
* ``GettingStarted/Examples/Circuits/quasar_pauli.ipynb``

.. rubric:: Classes

* :py:class:`~quasar.pauli.Pauli`
* :py:class:`~quasar.pauli.PauliExpectation`
* :py:class:`~quasar.pauli.PauliOperator`
* :py:class:`~quasar.pauli.PauliString`

Class Pauli
^^^^^^^^^^^

.. rubric:: Math

.. hlist:: 
    :columns: 7

    * :py:meth:`~quasar.pauli.Pauli.conj`
    * :py:meth:`~quasar.pauli.Pauli.dot`
    * :py:meth:`~quasar.pauli.Pauli.norm2`
    * :py:meth:`~quasar.pauli.Pauli.norminf`
    * :py:meth:`~quasar.pauli.Pauli.zero`
    * :py:meth:`~quasar.pauli.Pauli.zeros_like`
    * :py:meth:`~quasar.pauli.Pauli.sieved`

.. rubric:: Pauli Summary Attributes

.. hlist::
    :columns: 4

    * :py:meth:`~quasar.pauli.Pauli.max_order`
    * :py:meth:`~quasar.pauli.Pauli.max_qubit`
    * :py:meth:`~quasar.pauli.Pauli.min_qubit`
    * :py:meth:`~quasar.pauli.Pauli.nqubit`
    * :py:meth:`~quasar.pauli.Pauli.nqubit_sparse`
    * :py:meth:`~quasar.pauli.Pauli.nterm`
    * :py:meth:`~quasar.pauli.Pauli.qubits`
    * :py:meth:`~quasar.pauli.Pauli.summary_str`

.. rubric:: Hilbert Space Representation

.. hlist::
    :columns: 3

    * :py:meth:`~quasar.pauli.Pauli.from_matrix`
    * :py:meth:`~quasar.pauli.Pauli.matrix_vector_product`
    * :py:meth:`~quasar.pauli.Pauli.to_matrix`

.. rubric:: OrderedDict Methods

.. hlist::
    :columns: 3

    * :py:meth:`~quasar.pauli.Pauli.get`
    * :py:meth:`~quasar.pauli.Pauli.setdefault`
    * :py:meth:`~quasar.pauli.Pauli.update`

.. rubric:: Pauli Starter Objects

* :py:meth:`~quasar.pauli.Pauli.IXYZ`

Class PauliExpectation
^^^^^^^^^^^^^^^^^^^^^^

.. hlist::
    :columns: 2

    * :py:meth:`~quasar.pauli.PauliExpectation.zero`
    * :py:meth:`~quasar.pauli.PauliExpectation.zeros_like`

Class PauliOperator
^^^^^^^^^^^^^^^^^^^

.. hlist::
    :columns: 3

    * :py:meth:`~quasar.pauli.PauliOperator.char`
    * :py:meth:`~quasar.pauli.PauliOperator.from_string`
    * :py:meth:`~quasar.pauli.PauliOperator.qubit`

Class PauliString
^^^^^^^^^^^^^^^^^

.. hlist::
    :columns: 4

    * :py:meth:`~quasar.pauli.PauliString.chars`
    * :py:meth:`~quasar.pauli.PauliString.from_string`
    * :py:meth:`~quasar.pauli.PauliString.order`
    * :py:meth:`~quasar.pauli.PauliString.qubits`

.. automodule:: quasar.pauli
   :members:
