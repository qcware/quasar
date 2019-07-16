import quasar
import numpy as np
from util.error import L1_error


"""
Test "PauliOperator" Class
"""

def util_get_pauli_operator():
    """
    A utility function for the testing functions below.
    It creates an instance of PauliOperator().
    """
    p = quasar.PauliOperator(2, 'Y')

    return p


def qubit():
    """
    Validate qubit() in the "PauliOperator" class.
    """
    p = util_get_pauli_operator()

    return p.qubit == 2
    
    
def char():
    """
    Validate char() in the "PauliOperator" class.
    """
    p = util_get_pauli_operator()

    return p.char == 'Y'    
    

def str():
    """
    Validate char() in the "PauliOperator" class.
    """
    p = util_get_pauli_operator()

    return p.__str__() == 'Y2'      

    
def from_string():
    """
    Validate char() in the "PauliOperator" class.
    """
    p = quasar.PauliOperator.from_string('Z5')

    return p.char == 'Z' and p.qubit == 5  
    
    

"""
Test "PauliString" Class
"""

def util_get_pauli_string(operators_list=['Z4','Y3','X2','Z1']):
    """
    A utility function for the testing functions below.
    It creates an instance of PauliString().
    """
    operators = ()
    for op in operators_list:
        operators+=(quasar.PauliOperator.from_string(op),)
    ps = quasar.PauliString(operators)

    return ps


def order():
    """
    Validate order() in the "PauliString" class.
    """
    ps = util_get_pauli_string()
    
    return ps.order==4
    
    
def qubits():
    """
    Validate qubits() in the "PauliString" class.
    """
    ps = util_get_pauli_string()
    
    return ps.qubits==(4,3,2,1)


def chars():
    """
    Validate chars() in the "PauliString" class.
    """
    ps = util_get_pauli_string()
    
    return ps.chars==('Z', 'Y', 'X', 'Z')

    
def str():
    """
    Validate __str__() in the "PauliString" class.
    """
    ps = util_get_pauli_string()
    
    return ps.__str__()=='Z4*Y3*X2*Z1'


def from_string():
    """
    Validate from_string() in the "PauliString" class.
    """
    ps = quasar.PauliString.from_string('Z4*Y3*X2*Z1')
    
    return ps.__str__()=='Z4*Y3*X2*Z1'


def I():
    """
    Validate I() in the "PauliString" class.
    """
    ps = quasar.PauliString.I
    
    return ps.qubits==()


    
"""
Test "Pauli" Class
"""

def util_get_pauli():
    """
    A utility function for the testing functions below.
    It creates an instance of Pauli()
    """
    ps1 = util_get_pauli_string(operators_list=['Z4','Y3','X2','Z1'])
    ps2 = util_get_pauli_string(operators_list=['X0','Z3'])
    
    keys = [ps1, ps2]
    
    return quasar.Pauli({ps1:0.11, ps2:0.22}), keys


def contains():
    """
    Validate __contains__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    
    return keys[0] in pauli

    
def getitem():
    """
    Validate __getitem__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    
    return pauli[keys[0]] == 0.11   
    
    
def setitem():
    """
    Validate __setitem__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli[keys[1]] = 0.55
    
    return pauli[keys[0]] == 0.11 and pauli[keys[1]] == 0.55    
    
    
def get():
    """
    Validate get() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    
    return pauli.get(keys[0]) == 0.11     
    
    
def setdefault():
    """
    Validate setdefault() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli.pop(keys[0])
    pauli.setdefault(keys[0], 0.55)
    pauli.setdefault(keys[1], 0.55)
    
    return pauli[keys[0]]==0.55 and pauli[keys[1]]==0.22 
    
    
def str():
    """
    Validate __str__() in the "Pauli" class.
    """
    pauli, _ = util_get_pauli()
    
    return pauli.__str__().replace('\n', '')=='+0.11*Z4*Y3*X2*Z1+0.22*X0*Z3'
    
    
def summary_str():
    """
    Validate summary_str() in the "Pauli" class.
    """
    pauli, _ = util_get_pauli()
    
    return pauli.summary_str.replace('\n', '')=='Pauli:  N          = 5  nterm      = 2  max_order  = 4'
    
    
def N():
    """
    Validate N() in the "Pauli" class.
    """
    pauli, _ = util_get_pauli()
    
    return pauli.N==5
    
    
def nterm():
    """
    Validate nterm() in the "Pauli" class.
    """
    pauli, _ = util_get_pauli()
    
    return pauli.nterm==2


def max_order():
    """
    Validate max_order() in the "Pauli" class.
    """
    pauli, _ = util_get_pauli()
    
    return pauli.max_order==4

    
def pos():
    """
    Validate __pos__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli + pauli
    
    return pauli[keys[0]]==0.22 and pauli[keys[1]]==0.44


def neg():
    """
    Validate __neg__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli - pauli
    
    return pauli[keys[0]]==0.0 and pauli[keys[1]]==0.0
    

def mul():
    """
    Validate __mul__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli * pauli

    key1 = quasar.PauliString.I
    key2 = util_get_pauli_string(operators_list=['Z4','X3','X2','Z1','X0'])
    key3 = util_get_pauli_string(operators_list=['X0','X3','Z4','X2','Z1'])
    
    return pauli[key1]==0.0605 and pauli[key2]==0.0242*1j and pauli[key3]==-0.0242*1j
    
    
def rmul():
    """
    Validate __rmul__() in the "Pauli" class.
    """
    pauli, _ = util_get_pauli()
    
    return 2*pauli==pauli*2
    
    
def truediv():
    """
    Validate __truediv__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli / 11.0

    return pauli[keys[0]]==0.01 and pauli[keys[1]]==0.02    
    

def add():
    """
    Validate __add__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli + 5.0

    return pauli[quasar.PauliString.I]==5.0


def sub():
    """
    Validate __sub__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli - 5.0

    return pauli[quasar.PauliString.I]==-5.0


def radd():
    """
    Validate __radd__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()

    return pauli+5.0 == 5.0+pauli


def rsub():
    """
    Validate __rsub__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = 5.0 - pauli
    
    return pauli[keys[0]]==-0.11 and pauli[keys[1]]==-0.22 

    
def iadd():
    """
    Validate __iadd__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli + 5.0*1j
    
    return pauli[quasar.PauliString.I]==5.0*1j 


def isub():
    """
    Validate __isub__() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli - 5.0*1j
    
    return pauli[quasar.PauliString.I]==-5.0*1j 


def dot():
    """
    Validate dot() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    
    return pauli.dot(pauli)== 0.0605
    
    
def conj():
    """
    Validate conj() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli + 5.0*1j
    
    return pauli.conj[quasar.PauliString.I]==-5.0*1j    
    
    
def norm2():
    """
    Validate norm2() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    
    return L1_error([(pauli.norm2, np.sqrt(5)*0.11)])

    
def norminf():
    """
    Validate norminf() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    
    return pauli.norminf == max(pauli.values())


def zero():
    """
    Validate zero() in the "Pauli" class.
    """
    pauli = quasar.Pauli.zero()
    
    return len(pauli.items())==0


def zeros_like():
    """
    Validate zeros_like() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = quasar.Pauli.zeros_like(pauli)
    
    return pauli[keys[0]]==0.0 and pauli[keys[1]]==0.0
    
    
def sieved():
    """
    Validate sieved() in the "Pauli" class.
    """
    pauli, _ = util_get_pauli()
    pauli = pauli + 1e-15

    return not quasar.PauliString.I in pauli.sieved()
    
    
def I():
    """
    Validate I() in the "Pauli" class.
    """
    pauli = quasar.Pauli.I()

    return (pauli[quasar.PauliString.I]==1) and len(pauli)==1
    
    
def IXYZ():
    """
    Validate IXYZ() in the "Pauli" class.
    """
    pauli = quasar.Pauli.IXYZ()

    return len(pauli)==4
    
    
def extract_orders():
    """
    Validate extract_orders() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()
    pauli = pauli.extract_orders(2)

    return (not keys[0] in pauli) and (keys[1] in pauli)
    
    
def qubits():
    """
    Validate qubits() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()

    return pauli.qubits==((4, 3, 2, 1), (0, 3))
    
    
def chars():
    """
    Validate chars() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()

    return pauli.chars==(('Z', 'Y', 'X', 'Z'), ('X', 'Z'))
    
    
def unique_chars():
    """
    Validate unique_chars() in the "Pauli" class.
    """
    pauli, keys = util_get_pauli()

    return pauli.unique_chars==('X', 'Y', 'Z')   
    
    
def compute_hilbert_matrix():
    """
    Validate compute_hilbert_matrix() in the "Pauli" class.
    """
    ps = util_get_pauli_string(operators_list=['X0','Z1'])
    pauli = quasar.Pauli({ps:0.2})
    
    hilbert_matrix = pauli.compute_hilbert_matrix()
    ans = np.zeros((4,4))
    ans[0,2] =  0.2
    ans[1,3] = -0.2
    ans[2,0] =  0.2
    ans[3,1] = -0.2
    
    return L1_error((hilbert_matrix, ans))


def compute_hilbert_matrix_vector_product():
    """
    Validate compute_hilbert_matrix_vector_product() in the "Pauli" class.
    """
    ps = util_get_pauli_string(operators_list=['X0','Z1'])
    pauli = quasar.Pauli({ps:0.2})
    wfn = np.array([1,0,0,0], dtype=np.complex128)
    
    product = pauli.compute_hilbert_matrix_vector_product(wfn)
    ans = [0,0,0.2,0]

    return L1_error((product, ans))    
    
    
    
"""
Test "PauliStarter" Class
"""

def paulistarter():
    """
    Validate "PauliStarter" class.
    """
    pauli_X = quasar.PauliStarter('X')
    pauli = pauli_X[2]
    
    ps = util_get_pauli_string(operators_list=['X2'])
    pauli_ans = quasar.Pauli({ps:1.0})
    
    return pauli == pauli_ans


# """
# Test "PauliJordanWigner" Class
# """

# def pauli_pordan_wigner():
    # """
    # Validate "PauliJordanWigner" class.
    # """
    # pauliJW = quasar.PauliJordanWigner().Composition()
    
    # return pauliJW[1]
    
    
# print(pauli_pordan_wigner())    
    
    
    
    
    
    
    
    