import numpy as np
from .measurement import ProbabilityHistogram
    
class Algebra(object):

    @staticmethod
    def apply_operator_1(
        statevector1,
        statevector2,
        operator,
        A,
        ):

        """ Apply a 1-body operator to statevector1 at qubit A, yielding statevector2.

        The formal operation performed is,

            statevector1_LiR = \\sum_{j} operator_ij statevector2_LjR

        Here L are the indices of all of the qubits to the left of A (<A), and
        R are the indices of all of the qubits to the right of A (>A).

        This function requires the user to supply both the initial state in
        statevector1 and an array statevector2 to place the result into. This allows this
        function to apply the gate without any new allocations or scratch arrays.

        Params:
            statevector1 (np.ndarray of shape (2**N,) and a complex dtype)
                - the initial statevector. operatornaffected by the operation
            statevector2 (np.ndarray of shape (2**N,) and a complex dtype)
                - an array to write the new statevector into. Overwritten by
                the operation.
            operator (np.ndarray of shape (2,2) and a complex dtype) - the matrix
                representation of the 1-body gate.
            A (int) - the qubit index to apply the gate at.
        Result:
            the data of statevector2 is overwritten with the result of the operation.
        Returns:
            reference to statevector2, for chaining
        """

        N = (statevector1.shape[0]&-statevector1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if operator.shape != (2,2): raise RuntimeError('1-body gate must be (2,2)')
        if statevector1.shape != (2**N,): raise RuntimeError('statevector1 should be (%d,) shape, is %r shape' % (2**N, statevector1.shape))
        if statevector2.shape != (2**N,): raise RuntimeError('statevector2 should be (%d,) shape, is %r shape' % (2**N, statevector2.shape))

        L = 2**(A)     # Left hangover
        R = 2**(N-A-1) # Right hangover
        statevector1v = statevector1.view() 
        statevector2v = statevector2.view()
        statevector1v.shape = (L,2,R)
        statevector2v.shape = (L,2,R)
        np.einsum('LjR,ij->LiR', statevector1v, operator, out=statevector2v)

        return statevector2, statevector1

    @staticmethod
    def apply_operator_2(
        statevector1,
        statevector2,
        operator,
        A,
        B,
        ):

        """ Apply a 2-body operator to statevector1 at qubits A and B, yielding statevector2.

        The formal operation performed is (for the case that A < B),

            statevector1_LiMjR = \\sum_{lk} operator_ijkl statevector2_LiMjR

        Here L are the indices of all of the qubits to the left of A (<A), M M
        are the indices of all of the qubits to the right of A (>A) and left of
        B (<B), and R are the indices of all of the qubits to the right of B
        (>B). If A > B, permutations of A and B and the gate matrix operator are
        performed to ensure that the gate is applied correctly.

        This function requires the user to supply both the initial state in
        statevector1 and an array statevector2 to place the result into. This allows this
        function to apply the gate without any new allocations or scratch arrays.

        Params:
            statevector1 (np.ndarray of shape (2**N,) and a complex dtype)
                - the initial statevector. operatornaffected by the operation
            statevector2 (np.ndarray of shape (2**N,) and a complex dtype)
                - an array to write the new statevector into. Overwritten by
                the operation.
            operator (np.ndarray of shape (4,4) and a complex dtype) - the matrix
                representation of the 2-body gate. This should be packed to
                operate on the product state |A> otimes |B>, as usual.
            A (int) - the first qubit index to apply the gate at.
            B (int) - the second qubit index to apply the gate at.
        Result:
            the data of statevector2 is overwritten with the result of the operation.
        Returns:
            reference to statevector2, for chaining
        """

        N = (statevector1.shape[0]&-statevector1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if B >= N: raise RuntimeError('B >= N')
        if A == B: raise RuntimeError('A == B')
        if operator.shape != (4,4): raise RuntimeError('2-body gate must be (4,4)')
        if statevector1.shape != (2**N,): raise RuntimeError('statevector1 should be (%d,) shape, is %r shape' % (2**N, statevector1.shape))
        if statevector2.shape != (2**N,): raise RuntimeError('statevector2 should be (%d,) shape, is %r shape' % (2**N, statevector2.shape))

        operator2 = np.reshape(operator, (2,2,2,2))
        if A > B:
            A2, B2 = B, A
            operator2 = np.einsum('ijkl->jilk', operator2)
        else:
            A2, B2 = A, B

        L = 2**(A2)      # Left hangover
        M = 2**(B2-A2-1) # Middle hangover
        R = 2**(N-B2-1)  # Right hangover
        statevector1v = statevector1.view() 
        statevector2v = statevector2.view()
        statevector1v.shape = (L,2,M,2,R)
        statevector2v.shape = (L,2,M,2,R)
        np.einsum('LkMlR,ijkl->LiMjR', statevector1v, operator2, out=statevector2v)

        return statevector2, statevector1

    @staticmethod
    def apply_operator_3(
        statevector1,
        statevector2,
        operator,
        A,
        B,
        C,
        ):

        """ Apply a 3-body operator to statevector1 at qubits A, B, and C, yielding statevector2.

        This function requires the user to supply both the initial state in
        statevector1 and an array statevector2 to place the result into. This allows this
        function to apply the gate without any new allocations or scratch arrays.

        Params:
            statevector1 (np.ndarray of shape (2**N,) and a complex dtype)
                - the initial statevector. operatornaffected by the operation
            statevector2 (np.ndarray of shape (2**N,) and a complex dtype)
                - an array to write the new statevector into. Overwritten by
                the operation.
            operator (np.ndarray of shape (8,8) and a complex dtype) - the matrix
                representation of the 3-body gate. This should be packed to
                operate on the product state |A> otimes |B> otimes |C>, as
                usual.
            A (int) - the first qubit index to apply the gate at.
            B (int) - the second qubit index to apply the gate at.
            C (int) - the third qubit index to apply the gate at.
        Result:
            the data of statevector2 is overwritten with the result of the operation.
        Returns:
            reference to statevector2, for chaining
        """

        N = (statevector1.shape[0]&-statevector1.shape[0]).bit_length()-1
        if A >= N: raise RuntimeError('A >= N')
        if B >= N: raise RuntimeError('B >= N')
        if C >= N: raise RuntimeError('C >= N')
        if A == B: raise RuntimeError('A == B')
        if A == C: raise RuntimeError('A == C')
        if B == C: raise RuntimeError('B == C')
        if operator.shape != (8,8): raise RuntimeError('3-body gate must be (8,8)')
        if statevector1.shape != (2**N,): raise RuntimeError('statevector1 should be (%d,) shape, is %r shape' % (2**N, statevector1.shape))
        if statevector2.shape != (2**N,): raise RuntimeError('statevector2 should be (%d,) shape, is %r shape' % (2**N, statevector2.shape))

        A2, B2, C2 = sorted((A, B, C))

        operator2 = np.reshape(operator, (2,2,2,2,2,2))

        bra_indices = 'ijk'
        ket_indices = 'lmn'
        bra_indices2 = ''.join([bra_indices[(A, B, C).index(_)] for _ in (A2, B2, C2)])
        ket_indices2 = ''.join([ket_indices[(A, B, C).index(_)] for _ in (A2, B2, C2)])
        
        operator2 = np.einsum('%s%s->%s%s' % (bra_indices, ket_indices, bra_indices2, ket_indices2), operator2)

        L = 2**(A2)      # Left hangover
        M = 2**(B2-A2-1) # Middle1 hangover
        P = 2**(C2-B2-1) # Middle2 hangover
        R = 2**(N-C2-1)  # Right hangover
        statevector1v = statevector1.view() 
        statevector2v = statevector2.view()
        statevector1v.shape = (L,2,M,2,P,2,R)
        statevector2v.shape = (L,2,M,2,P,2,R)
        np.einsum('LlMmPnR,ijklmn->LiMjPkR', statevector1v, operator2, out=statevector2v)

        return statevector2, statevector1
    
    @staticmethod
    def apply_operator_n(
        statevector1,
        statevector2,
        operator,
        qubits,
        ):

        N = (statevector1.shape[0]&-statevector1.shape[0]).bit_length()-1
        if any(_ >= N for _ in qubits): raise RuntimeError('qubits >= N')
        if len(set(qubits)) != len(qubits): raise RuntimeError('duplicate entry in qubits')
        if statevector1.shape != (2**N,): raise RuntimeError('statevector1 should be (%d,) shape, is %r shape' % (2**N, statevector1.shape))
        if statevector2.shape != (2**N,): raise RuntimeError('statevector2 should be (%d,) shape, is %r shape' % (2**N, statevector2.shape))
           
        # hangover
        qubits2 = tuple(sorted(qubits))
        hangovers = (2**qubits2[0],) + tuple(2**(qubits2[A+1]-qubits2[A]-1) for A in range(len(qubits2)-1)) + (2**(N-qubits2[-1]-1),)
        shape = []
        for hangover in hangovers[:-1]:
            shape.append(hangover)
            shape.append(2)
        shape.append(hangovers[-1])
        shape = tuple(shape)
        
        statevector1v = statevector1.view() 
        statevector2v = statevector2.view()
        statevector1v.shape = shape
        statevector2v.shape = shape
        
        # symbol stock 
        hangover_stock = 'ABCDEFGHIJKLMNOPQRSToperatorVWXYZ'
        bra_stock = 'abcdefghijklm'
        ket_stock = 'nopqrstuvwxyz'
    
        M = len(qubits)
        if M > 13: raise RuntimeError('Technical limit: cannot run N > 13')
        
        # einsum form for ordering gate
        bra_str = bra_stock[:M]
        ket_str = ket_stock[:M]
        bra_str2 = ''.join([bra_str[qubits.index(_)] for _ in qubits2])
        ket_str2 = ''.join([ket_str[qubits.index(_)] for _ in qubits2])
        operator_str = ket_stock[:M] + bra_stock[:M]
        
        shape_operator = tuple(2 for _ in range(2*M))
        operator2 = np.reshape(operator, shape_operator)
        operator2 = np.einsum('%s%s->%s%s' % (bra_str, ket_str, bra_str2, ket_str2), operator2)
        
        # einsum form for applying gate
        statevector1_str = ''
        statevector2_str = ''
        for A in range(M):
            statevector1_str += hangover_stock[A]
            statevector1_str += bra_stock[A]
            statevector2_str += hangover_stock[A]
            statevector2_str += ket_stock[A]
        statevector1_str += hangover_stock[M]
        statevector2_str += hangover_stock[M]
        
        np.einsum('%s,%s->%s' % (statevector1_str, operator_str, statevector2_str), statevector1v, operator2, out=statevector2v)
        statevector2v = np.reshape(statevector2v, (-1,))
        
        return statevector2, statevector1

    @staticmethod
    def sample_histogram_from_probabilities(
        probabilities,
        nmeasurement=None,
        cutoff=1.0E-12,
        ):

        """ Randomly sample binary string measurements from a set of
            probabilities of size 2**N.

        Params:
            statevector (np.ndarray of shape (2**N,)) - the statevector to
                sample.
            nmeasurement (int or None) - number of measurements to sample. If
                None, infinite sampling is assumed and the probabilities are
                directly returned in ProbabilityHistogram format.
            cutoff (float) - probability below which to ignore contributions.
        Returns:
            (ProbabilityHistogram) - a ProbabilityHistogram object containing
                the results of randomly sampled projective measurements.
        """

        N = (probabilities.shape[0]&-probabilities.shape[0]).bit_length()-1
    
        # Directly return probabilities if nmeasurement is None (infinite sampling)
        if nmeasurement is None:
            return ProbabilityHistogram(
                nqubit=N,
                histogram={ k : v for k, v in enumerate(probabilities) if v > cutoff},
                nmeasurement=nmeasurement,
                ) 
        # Otherwise, perform random sampling
        if not isinstance(nmeasurement, int):
            raise RuntimeError('nmeasurement must be int: %s' % nmeasurement)
        I = list(np.searchsorted(np.cumsum(probabilities), np.random.rand(nmeasurement)))
        return ProbabilityHistogram(
            nqubit=N,
            histogram={ int(k) : I.count(k) / nmeasurement for k in list(sorted(set(I))) },
            nmeasurement=nmeasurement,
            ) 
