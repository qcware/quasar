import unittest
import simple_test
import test_circuit, test_matrix, test_ket, test_MeasurementResult, test_OptimizationResult, test_Pauli


class Test(unittest.TestCase):
    def test_test(self):
        # should always be true
        self.assertTrue(simple_test.test())
    def test_test1(self, one=1):
        # First statement should return True. Second should return False.
        self.assertTrue(simple_test.test1(1))
        # self.assertTrue(simple_test.test1(2))
    
    # ==> Test Circuit class in circuit.py <==
    def test_circuit_simulate(self):
        self.assertTrue(test_circuit.simulate())
    def test_circuit_simulate_steps(self):
        self.assertTrue(test_circuit.simulate_steps())
    def test_circuit_apply_gate_1(self):     
        self.assertTrue(test_circuit.apply_gate_1())
        self.assertTrue(test_circuit.apply_gate_1_format())
    def test_circuit_apply_gate_2(self):
        self.assertTrue(test_circuit.apply_gate_2())
        self.assertTrue(test_circuit.apply_gate_2_format())
    def test_circuit_apply_gate_3(self):
        self.assertTrue(test_circuit.apply_gate_3())
        self.assertTrue(test_circuit.apply_gate_3_format())
    def test_circuit_compute_1pdm(self):
        self.assertTrue(test_circuit.compute_1pdm())
        self.assertTrue(test_circuit.compute_1pdm_format())
    def test_circuit_compute_2pdm(self):
        self.assertTrue(test_circuit.compute_2pdm())
        self.assertTrue(test_circuit.compute_2pdm_format())
    def test_circuit_compute_3pdm(self):
        self.assertTrue(test_circuit.compute_3pdm())
        self.assertTrue(test_circuit.compute_3pdm_format())
    def test_circuit_compute_4pdm(self):
        self.assertTrue(test_circuit.compute_4pdm())
        self.assertTrue(test_circuit.compute_4pdm_format())
    def test_circuit_compute_npdm(self):
        self.assertTrue(test_circuit.compute_npdm())
        self.assertTrue(test_circuit.compute_npdm_format())
    def test_circuit_compute_pauli_1(self):
        self.assertTrue(test_circuit.compute_pauli_1())
    def test_circuit_compute_pauli_2(self):
        self.assertTrue(test_circuit.compute_pauli_2())
    def test_circuit_compute_pauli_3(self):
        self.assertTrue(test_circuit.compute_pauli_3())
    def test_circuit_compute_pauli_4(self):
        self.assertTrue(test_circuit.compute_pauli_4())
    def test_circuit_compute_pauli_n(self):
        self.assertTrue(test_circuit.compute_pauli_n())
    def test_circuit_measure(self):
        self.assertTrue(test_circuit.measure())
    def test_circuit_compute_measurements_from_statevector(self):
        self.assertTrue(test_circuit.compute_measurements_from_statevector())
    def test_circuit_nparam(self):
        self.assertTrue(test_circuit.nparam())
    def test_circuit_param_keys(self):
        self.assertTrue(test_circuit.param_keys())
    def test_circuit_param_values(self):
        self.assertTrue(test_circuit.param_values())
    def test_circuit_set_param_values(self):
        self.assertTrue(test_circuit.set_param_values())
    def test_circuit_params(self):
        self.assertTrue(test_circuit.params())    
    def test_circuit_set_param(self):
        self.assertTrue(test_circuit.set_param())
    def test_circuit_param_str(self):
        self.assertTrue(test_circuit.param_str())    
    def test_circuit_I(self):
        self.assertTrue(test_circuit.I())      
    def test_circuit_X(self):
        self.assertTrue(test_circuit.X()) 
    def test_circuit_Y(self):
        self.assertTrue(test_circuit.Y())      
    def test_circuit_Z(self):
        self.assertTrue(test_circuit.Z())     
    def test_circuit_H(self):
        self.assertTrue(test_circuit.H())  
    def test_circuit_S(self):
        self.assertTrue(test_circuit.S()) 
    def test_circuit_T(self):
        self.assertTrue(test_circuit.T()) 
    def test_circuit_Rx2(self):
        self.assertTrue(test_circuit.Rx2()) 
    def test_circuit_Rx2T(self):
        self.assertTrue(test_circuit.Rx2T()) 
    def test_circuit_CX(self):
        self.assertTrue(test_circuit.CX())
    def test_circuit_CY(self):
        self.assertTrue(test_circuit.CY())
    def test_circuit_CZ(self):
        self.assertTrue(test_circuit.CZ())        
    def test_circuit_CS(self):
        self.assertTrue(test_circuit.CS())        
    def test_circuit_SWAP(self):
        self.assertTrue(test_circuit.SWAP())
    def test_circuit_CCX(self):
        self.assertTrue(test_circuit.CCX())        
    def test_circuit_CSWAP(self):
        self.assertTrue(test_circuit.CSWAP())        
    def test_circuit_Rx(self):
        self.assertTrue(test_circuit.Rx()) 
    def test_circuit_Ry(self):
        self.assertTrue(test_circuit.Ry())         
    def test_circuit_Rz(self):
        self.assertTrue(test_circuit.Rz())         
    def test_circuit_u1(self):
        self.assertTrue(test_circuit.u1())         
    def test_circuit_u2(self):
        self.assertTrue(test_circuit.u2())         
    def test_circuit_u3(self):
        self.assertTrue(test_circuit.u3())      
    def test_circuit_SO4(self):
        self.assertTrue(test_circuit.SO4())          
    def test_circuit_SO42(self):
        self.assertTrue(test_circuit.SO42())          
    def test_circuit_CF(self):
        self.assertTrue(test_circuit.CF()) 
    def test_circuit_R_ion(self):
        self.assertTrue(test_circuit.R_ion())        
    def test_circuit_Rx_ion(self):
        self.assertTrue(test_circuit.Rx_ion()) 
    def test_circuit_Ry_ion(self):
        self.assertTrue(test_circuit.Ry_ion()) 
    def test_circuit_Rz_ion(self):
        self.assertTrue(test_circuit.Rz_ion()) 
    def test_circuit_XX_ion(self):
        self.assertTrue(test_circuit.XX_ion())         
    def test_circuit_U1(self):
        self.assertTrue(test_circuit.U1())          
    def test_circuit_U2(self):
        self.assertTrue(test_circuit.U2())         
        

    # ==> Test Matrix class in circuit.py <==
    def test_matrix_1qubit_constant_matrices(self):
        self.assertTrue(test_matrix.one_qubit_constant_matrices())
    def test_matrix_2qubit_constant_matrices(self):
        self.assertTrue(test_matrix.two_qubit_constant_matrices())
    def test_matrix_3qubit_constant_matrices(self):
        self.assertTrue(test_matrix.three_qubit_constant_matrices())
    def test_matrix_1qubit_1param_matrices(self):
        self.assertTrue(test_matrix.one_qubit_1param_matrices())
    def test_matrix_1qubit_2param_matrices(self):
        self.assertTrue(test_matrix.one_qubit_2param_matrices())
    def test_matrix_1qubit_2param_matrices(self):
        self.assertTrue(test_matrix.one_qubit_3param_matrices())
    def test_matrix_2qubit_1param_matrices(self):
        self.assertTrue(test_matrix.two_qubit_1param_matrices())


    # ==> Test Ket class in measurement.py <==
    def test_ket_initialization(self):
        self.assertTrue(test_ket.initialization())
    def test_ket_getitem(self):
        self.assertTrue(test_ket.getitem())
    def test_ket_N(self):
        self.assertTrue(test_ket.N())
    def test_ket_from_int(self):
        self.assertTrue(test_ket.from_int())


    # ==> Test MeasurementResult class in measurement.py <==
    def test_MeasurementResult_initialization(self):
        self.assertTrue(test_MeasurementResult.initialization())
    def test_MeasurementResult_contains(self):
        self.assertTrue(test_MeasurementResult.contains())
    def test_MeasurementResult_getitem(self):
        self.assertTrue(test_MeasurementResult.getitem())
    def test_MeasurementResult_setitem(self):
        self.assertTrue(test_MeasurementResult.setitem())
    def test_MeasurementResult_get(self):
        self.assertTrue(test_MeasurementResult.get())
    def test_MeasurementResult_setdefault(self):
        self.assertTrue(test_MeasurementResult.setdefault())
    def test_MeasurementResult_N(self):
        self.assertTrue(test_MeasurementResult.N())
    def test_MeasurementResult_nmeasurement(self):
        self.assertTrue(test_MeasurementResult.nmeasurement())
    def test_MeasurementResult_str(self):
        self.assertTrue(test_MeasurementResult.str())
    def test_MeasurementResult_subset(self):
        self.assertTrue(test_MeasurementResult.subset())


    # ==> Test OptimizationResult class in measurement.py <==
    def test_OptimizationResult_initialization(self):
        self.assertTrue(test_OptimizationResult.initialization())
    def test_OptimizationResult_contains(self):
        self.assertTrue(test_OptimizationResult.contains())
    def test_OptimizationResult_getitem(self):
        self.assertTrue(test_OptimizationResult.getitem())
    def test_OptimizationResult_setitem(self):
        self.assertTrue(test_OptimizationResult.setitem())
    def test_OptimizationResult_get(self):
        self.assertTrue(test_OptimizationResult.get())
    def test_OptimizationResult_setdefault(self):
        self.assertTrue(test_OptimizationResult.setdefault())
    def test_OptimizationResult_N(self):
        self.assertTrue(test_OptimizationResult.N())
    def test_OptimizationResult_str(self):
        self.assertTrue(test_OptimizationResult.str())
    def test_OptimizationResult_energy_sorted(self):
        self.assertTrue(test_OptimizationResult.energy_sorted())
    def test_OptimizationResult_merge(self):
        self.assertTrue(test_OptimizationResult.merge())

        
    # ==> Test PauliOperator class in pauli.py <==
    def test_PauliOperator_qubit(self):
        self.assertTrue(test_Pauli.qubit())
    def test_PauliOperator_char(self):
        self.assertTrue(test_Pauli.char())
    def test_PauliOperator_str(self):
        self.assertTrue(test_Pauli.str())
    def test_PauliOperator_from_string(self):
        self.assertTrue(test_Pauli.from_string())

        
    # ==> Test PauliString class in pauli.py <==
    def test_PauliString_order(self):
        self.assertTrue(test_Pauli.order())
    def test_PauliString_qubits(self):
        self.assertTrue(test_Pauli.qubits())
    def test_PauliString_chars(self):
        self.assertTrue(test_Pauli.chars())
    def test_PauliString_str(self):
        self.assertTrue(test_Pauli.str())        
    def test_PauliString_from_string(self):
        self.assertTrue(test_Pauli.from_string())
    def test_PauliString_I(self):
        self.assertTrue(test_Pauli.I())         
        
        
    # ==> Test Pauli class in pauli.py <==
    def test_Pauli_contains(self):
        self.assertTrue(test_Pauli.contains())
    def test_Pauli_getitem(self):
        self.assertTrue(test_Pauli.getitem())
    def test_Pauli_setitem(self):
        self.assertTrue(test_Pauli.setitem())
    def test_Pauli_get(self):
        self.assertTrue(test_Pauli.get())        
    def test_Pauli_setdefault(self):
        self.assertTrue(test_Pauli.setdefault())
    def test_Pauli_str(self):
        self.assertTrue(test_Pauli.str())
    def test_Pauli_summary_str(self):
        self.assertTrue(test_Pauli.summary_str())
    def test_Pauli_N(self):
        self.assertTrue(test_Pauli.N())
    def test_Pauli_nterm(self):
        self.assertTrue(test_Pauli.nterm())        
    def test_Pauli_max_order(self):
        self.assertTrue(test_Pauli.max_order())
    def test_Pauli_pos(self):
        self.assertTrue(test_Pauli.pos())
    def test_Pauli_neg(self):
        self.assertTrue(test_Pauli.neg())
    def test_Pauli_mul(self):
        self.assertTrue(test_Pauli.mul())
    def test_Pauli_rmul(self):
        self.assertTrue(test_Pauli.rmul())        
    def test_Pauli_truediv(self):
        self.assertTrue(test_Pauli.truediv())        
    def test_Pauli_add(self):
        self.assertTrue(test_Pauli.add())
    def test_Pauli_sub(self):
        self.assertTrue(test_Pauli.sub())
    def test_Pauli_radd(self):
        self.assertTrue(test_Pauli.radd())
    def test_Pauli_rsub(self):
        self.assertTrue(test_Pauli.rsub())        
    def test_Pauli_iadd(self):
        self.assertTrue(test_Pauli.iadd())
    def test_Pauli_isub(self):
        self.assertTrue(test_Pauli.isub())
    def test_Pauli_dot(self):
        self.assertTrue(test_Pauli.dot())
    def test_Pauli_conj(self):
        self.assertTrue(test_Pauli.conj())
    def test_Pauli_norm2(self):
        self.assertTrue(test_Pauli.norm2())        
    def test_Pauli_norminf(self):
        self.assertTrue(test_Pauli.norminf())
    def test_Pauli_zero(self):
        self.assertTrue(test_Pauli.zero())
    def test_Pauli_zeros_like(self):
        self.assertTrue(test_Pauli.zeros_like())
    def test_Pauli_sieved(self):
        self.assertTrue(test_Pauli.sieved())
    def test_Pauli_I(self):
        self.assertTrue(test_Pauli.I())        
    def test_Pauli_IXYZ(self):
        self.assertTrue(test_Pauli.IXYZ())  
    def test_Pauli_extract_orders(self):
        self.assertTrue(test_Pauli.extract_orders())
    def test_Pauli_qubits(self):
        self.assertTrue(test_Pauli.qubits())  
    def test_Pauli_chars(self):
        self.assertTrue(test_Pauli.chars())        
    def test_Pauli_unique_chars(self):
        self.assertTrue(test_Pauli.unique_chars())  
    def test_Pauli_compute_hilbert_matrix(self):
        self.assertTrue(test_Pauli.compute_hilbert_matrix())        
    def test_Pauli_compute_hilbert_matrix_vector_product(self):
        self.assertTrue(test_Pauli.compute_hilbert_matrix_vector_product())  


    # ==> Test PauliStarter class in pauli.py <==
    def test_PauliStarter_paulistarter(self):
        self.assertTrue(test_Pauli.paulistarter())

        
        
if __name__ == '__main__':
    unittest.main()

    
    
    
    
    
    
    
    
    
    