# Quasar

Autotests: ![CircleCI](https://circleci.com/gh/qcware/quasar/tree/mark2.svg?style=svg&circle-token=e85544db6236d5ecb720ac042a9a40d2f819a4ec)

## Why Quasar

There are three key reasons that `quasar` might prove useful:
 * If you write your code in `quasar`, it will run in `qiskit` (IBM), `cirq` (Google), `pyquil` (Rigetti), `Q#` (Microsoft), and on IonQ's API.
 * If you write your code in `quasar`, you can easily access key high-level quantum primitives like Pauli expectations, parameter gradients, and parameter tomography.
 * If you write your code in `quasar`, it might in certain cases run considerably faster than in other quantum languages/implementations, due to some special techniques that we have baked into the library stack. 

## Supported Backends

 * `qiskit` (IBM): 0.10.0
 * `cirq` (Google): 0.6.0
 * `pyquil` (Rigetti): 2.13.0
 * `ionq` (IonQ): Beta version of REST web API
