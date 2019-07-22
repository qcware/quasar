Circuit
======================

A :py:class:`~quaar.Circuit` is effectively a list of gates and instructions which
can be created using the function documented in this section::

    >>> import quasar
    >>> 
    >>> circuit = quasar.Circuit(N=2)
    >>> circuit = circuit.X(0).CX(0,1)
        ...


Quasar Circuit() class
-----------------------------
.. automodule:: quasar.Circuit

