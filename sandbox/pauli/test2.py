import quasar
    
string = quasar.PauliString.from_string('X1*X2') 
string2 = quasar.PauliString.from_string('X1*X3') 

pauli = quasar.Pauli({
    string : 1.0,
    })

# print(pauli)
print(len(pauli))
print(pauli.values())
print(pauli.keys())

pauli[string] += 2.0
print(pauli[string])

# pauli.update(pauli)

print(pauli.get(string))

print(dir(pauli))
print(pauli)

print(pauli.get(string2, 4.0))
print(pauli.summary_str)

print(-pauli)
