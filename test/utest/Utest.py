import unittest
import test_circuit

class Test(unittest.TestCase):
   
    # ==> Tests for circuit class (ordered from top to bottom) <==
    def test_init_circuit(self):
        self.assertTrue(some_tests.init_circuit())
    def test_ntime(self):
        self.assertTrue(some_tests.ntime())
    def test_ngate(self):
        self.assertTrue(some_tests.ngate())
    def test_ngate1(self):
        self.assertTrue(some_tests.ngate1())
    def test_ngate2(self):
        self.assertTrue(some_tests.ngate2())
    def test_add_gate(self):
        self.assertTrue(some_tests.add_gate())
    def test_gate(self):
        self.assertTrue(some_tests.gate())
    def test_copy(self):
        self.assertTrue(some_tests.copy())
    def test_subset(self):
        self.assertTrue(some_tests.subset())
    def test_concatenate(self):
        self.assertTrue(some_tests.concatenate())
    def test_deadjoin(self):
        self.assertTrue(some_tests.deadjoin())
    def test_adjoin(self):
        self.assertTrue(some_tests.adjoin())
    def test_reversed(self):
        self.assertTrue(some_tests.test_reversed())
    def test_is_equivalent(self):
        self.assertTrue(some_tests.is_equivalent())
    def test_nonredundant(self):
        self.assertTrue(some_tests.nonredundant())
    def test_compressed(self):
        self.assertTrue(some_tests.compressed())
    def test_subcircuit(self):
        self.assertTrue(some_tests.subcircuit())
    def test_add_circuit(self):
        self.assertTrue(some_tests.add_circuit())
    def test_sort_gates(self):
        self.assertTrue(some_tests.sort_gates())
    def test_is_equivalent_order(self):
        self.assertTrue(some_tests.is_equivalent_order())
    # ==> Test Circuit class in circuit.py (ordered from bottom to top) <==
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

if __name__ == '__main__':
    unittest.main()
