import cirq

circuit = cirq.Circuit()

q0, q1 = cirq.LineQubit.range(2)

#circuit.append(q0, q1)

circuit.append(cirq.H(q0))

circuit.append(cirq.CNOT(q0, q1))

s = cirq.Simulator()

print('Simulate the circuit:')
results=s.simulate(circuit)
print(results)
print()

# For sampling, we need to add a measurement at the end bell_circuit.append(cirq.measure(q0, q1, key='result'))
print('Sample the circuit:')
samples=s.run(circuit, repetitions=1000) # Print a histogram of results print(samples.histogram(key='result'))
