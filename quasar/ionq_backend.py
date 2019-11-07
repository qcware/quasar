import numpy as np
from .backend import Backend
from .circuit import Circuit
from .algebra import Algebra
import json
import requests

class IonQBackend(Backend):

    api_url = 'https://api.ionq.co/v0'

    def __init__(
        self,
        api_key,
        ):

        self.api_key = api_key

    def __str__(self):
        return 'IonQBackend'

    @property
    def summary_str(self):
        s = 'IonQBackend\n'  
        s += '  %-7s = %-10s\n' % ('api_key', self.api_key)
        return s

    @property
    def has_run_statevector(self):
        return False

    @property
    def has_run_pauli_sigma(self):
        return False

    @property
    def has_statevector_input(self):
        return False

    def run_measurement(
        self,
        circuit,
        nmeasurement=None,
        statevector=None,
        dtype=np.complex128,
        **kwargs):
    
        if statevector is not None:
            raise RuntimeError('statevector input not supported')

        # => Post Job <= #

        url = IonQBackend.api_url + '/jobs'

        headers = {
            'Authorization' : 'apiKey %s' % self.api_key,
            'Content-Type' : 'application/json'
        }

        data = json.dumps(self.job_json_dict(
            circuit=circuit, 
            # nmeasurement=nmeasurement,
            nmeasurement=None,
            ))

        ret = requests.post(
            url,
            headers=headers,    
            data=data,
            )

        status = json.loads(ret.text)

        # TODO: Check for 401 error, etc

        job_id = status['id']

        # => Get Job Results <= #
        
        url = IonQBackend.api_url + '/jobs/' + job_id

        headers = {
            'Authorization' : 'apiKey %s' % self.api_key,
        } 

        ret = requests.get(
            url,
            headers=headers,    
            )

        status = json.loads(ret.text)
    
        histogram = status['data']['histogram']

        P = np.zeros((2**circuit.nqubit,), dtype=np.float64)
        for key, value in histogram.items():
            P[int(key)] = value

        # TODO: Check for 401 error, etc

        # TODO: Poll until 'status' 'completed'

        # TODO: Figure out uniform measurement sampling scheme (qpu vs simulator)

        return Algebra.sample_histogram_from_probabilities(P, nmeasurement)
        
    def job_json_dict(
        self,
        circuit,
        nmeasurement=None,
        ):

        body_dict = {}
        body_dict['lang'] = 'json'
        body_dict['body'] = IonQBackend.circuit_to_json_dict(circuit)
        if nmeasurement is None:
            body_dict['target'] = 'simulator'
        if nmeasurement is not None:
            body_dict['shots'] = nmeasurement
            body_dict['target'] = 'qpu'

        return body_dict

    @staticmethod
    def circuit_to_json_dict(
        circuit,
        ):

        if not isinstance(circuit, Circuit):
            raise RuntimeError('circuit must be of type Circuit')

        # Circuit gates
        circuit_list = []
        one_qubit_gate_names = {
            'X' : 'x',
            'Y' : 'y',
            'Z' : 'z',
            'H' : 'h', 
            'S' : 's',
            'T' : 't',
            'ST' : 'si',
            'TT' : 'ti',
        }
        min_qubit = circuit.min_qubit
        nqubit = circuit.nqubit
        for key, gate in circuit.gates.items():
            times, qubits = key
            if gate.name == 'I': 
                continue # No I gate in IonQ
            elif gate.nqubit == 1 and gate.name in one_qubit_gate_names:
                circuit_list.append({ 
                    'gate'    : one_qubit_gate_names[gate.name],
                    'target'  : qubits[0] - min_qubit,
                    })
            elif gate.nqubit == 1 and gate.name in ['Rx', 'Ry', 'Rz']:
                circuit_list.append({ 
                    'gate'     : gate.name.lower(),
                    'target'   : qubits[0] - min_qubit,
                    'rotation' : IonQBackend.quasar_to_ionq_angle(gate.parameters['theta']),
                    })
            elif gate.nqubit == 2 and gate.name == 'CX':
                circuit_list.append({ 
                    'gate'    : 'cnot',
                    'control' : qubits[0] - min_qubit,
                    'target'  : qubits[1] - min_qubit,
                    })
            else:
                raise RuntimeError('Unknown gate for IonQ: %s' % gate)

        json_dict = {}
        json_dict['qubits'] = nqubit
        json_dict['circuit'] = circuit_list
        
        return json_dict
        
    @staticmethod
    def quasar_to_ionq_angle(theta):
        return 2.0 * theta

    @staticmethod
    def ionq_to_quasar_angle(theta):
        return 0.5 * theta

