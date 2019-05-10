import quasar
import qiskit

qiskit.IBMQ.enable_account("eda5c2c406984b6244e23e2f554866294ce4c8639339f3a2eba0ddb96f13be8baf4715045b27f984d0a58745b678208eefc3ef9262fd1ff2674872dea7ee4846", "https://q-console-api.mybluemix.net/api/Hubs/ibmq/Groups/qc-ware/Projects/default")

backend = quasar.QiskitHardwareBackend('ibmq_20_tokyo')
# backend = quasar.QiskitHardwareBackend('ibmq_poughkeepsie')
print(backend)
