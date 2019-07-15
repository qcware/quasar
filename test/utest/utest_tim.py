import unittest
import simple_test
import test_circuit, test_matrix


class Test(unittest.TestCase):
    def test_test(self):
        # should always be true
        self.assertTrue(simple_test.test())
    def test_test1(self, one=1):
        # First statement should return True. Second should return False.
        self.assertTrue(simple_test.test1(1))
        # self.assertTrue(simple_test.test1(2))
    
    # ==> Test Circuit class <==
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
        

    # ==> Test Matrix class <==
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











        
if __name__ == '__main__':
    unittest.main()

    
    
    
    
    
    
    
    
    
    