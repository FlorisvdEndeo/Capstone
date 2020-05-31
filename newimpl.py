from copy import deepcopy
import math

import scipy.sparse.linalg as spsl

import projectq
from projectq import MainEngine
from projectq.backends._sim._simulator import Simulator
from projectq.meta import Compute, Control, Dagger, Uncompute
from projectq.ops import All, H, Measure, Ph, QubitOperator, R, StatePreparation, X, Z


num_qubits = 2

def get_eigenvalue_and_eigenvector(n_sites, hamiltonian, k, which='SA'):
    """
    Returns k eigenvalues and eigenvectors of the hamiltonian.

    Args:
        n_sites(int): Number of qubits/sites in the hamiltonian
        hamiltonian(QubitOperator): QubitOperator representating the Hamiltonian
        k: num of eigenvalue and eigenvector pairs (see spsl.eigsh k)
        which: see spsl.eigsh which

    """
    def mv(v):
        eng = projectq.MainEngine(backend=Simulator(), engine_list=[])
        qureg = eng.allocate_qureg(n_sites)
        eng.flush()
        eng.backend.set_wavefunction(v, qureg)
        eng.backend.apply_qubit_operator(hamiltonian, qureg)
        order, output = deepcopy(eng.backend.cheat())
        for i in order:
            assert i == order[i]
        eng.backend.set_wavefunction([1]+[0]*(2**n_sites-1), qureg)
        return output

    A = spsl.LinearOperator((2**n_sites,2**n_sites), matvec=mv)

    eigenvalues, eigenvectormatrix = spsl.eigsh(A, k=k, which=which)
    eigenvectors = []
    for i in range(k):
        eigenvectors.append(list(eigenvectormatrix[:, i]))
    return eigenvalues, eigenvectors

def W(eng, individual_terms, initial_wavefunction, ancilla_qubits, system_qubits):
    """
    Applies the W operator as defined in arXiv:1711.11025.

    Args:
        eng(MainEngine): compiler engine
        individual_terms(list<QubitOperator>): list of individual unitary
                                               QubitOperators. It applies
                                               individual_terms[0] if ancilla
                                               qubits are in state |0> where
                                               ancilla_qubits[0] is the least
                                               significant bit.
        initial_wavefunction: Initial wavefunction of the ancilla qubits
        ancilla_qubits(Qureg): ancilla quantum register in state |0>
        system_qubits(Qureg): system quantum register
    """
    # Apply V:
    for ancilla_state in range(len(individual_terms)):
        with Compute(eng):
            for bit_pos in range(len(ancilla_qubits)):
                if not (ancilla_state >> bit_pos) & 1:
                    X | ancilla_qubits[bit_pos]
        with Control(eng, ancilla_qubits):
            individual_terms[ancilla_state] | system_qubits
        Uncompute(eng)
    # Apply S: 1) Apply B^dagger
    with Compute(eng):
        with Dagger(eng):
            StatePreparation(initial_wavefunction) | ancilla_qubits
    # Apply S: 2) Apply I-2|0><0|
    with Compute(eng):
        All(X) | ancilla_qubits
    with Control(eng, ancilla_qubits[:-1]):
        Z | ancilla_qubits[-1]
    Uncompute(eng)
    # Apply S: 3) Apply B
    Uncompute(eng)
    # Could also be omitted and added when calculating the eigenvalues:
    Ph(math.pi) | system_qubits[0]

eng = projectq.MainEngine()
system_qubits = eng.allocate_qureg(num_qubits)

coefs = [[1.05, -0.562600, -0.248783, 0.199984,  0.00850998],[0.40, 0.460364, -0.688819, 0.164515, 0.0129140], [0.45, 0.267547, -0.633890, 0.1666621, 0.0127192]]

# \begin{array}{ccccc}
# 0.40 & 4.60364 \mathrm{E}-01 & -6.88819 \mathrm{E}-01 & 1.64515 \mathrm{E}-01 & 1.29140 \mathrm{E}-02 \\
# 0.45 & 2.67547 \mathrm{E}-01 & -6.33890 \mathrm{E}-01 & 1.66621 \mathrm{E}-01 & 1.27192 \mathrm{E}-02 \\
# 0.50 & 1.10647 \mathrm{E}-01 & -5.83080 \mathrm{E}-01 & 1.68870 \mathrm{E}-01 & 1.25165 \mathrm{E}-02 \\
# 0.55 & -1.83734 \mathrm{E}-02 & -5.36489 \mathrm{E}-01 & 1.71244 \mathrm{E}-01 & 1.23003 \mathrm{E}-02 \\
# 0.65 & -2.13932 \mathrm{E}-01 & -4.55433 \mathrm{E}-01 & 1.76318 \mathrm{E}-01 & 1.18019 \mathrm{E}-02 \\
# 0.75 & -3.49833 \mathrm{E}-01 & -3.88748 \mathrm{E}-01 & 1.81771 \mathrm{E}-01 & 1.11772 \mathrm{E}-02
# \end{array}



lstofham = []

for set in coefs:
    hamiltonian = QubitOperator()
    hamiltonian += QubitOperator("", set[1])
    hamiltonian += QubitOperator("Z0",  set[2])
    hamiltonian += QubitOperator("Z1", set[2])
    hamiltonian += QubitOperator("X0 X1", set[3])
    hamiltonian += QubitOperator("Z0 Z1", set[4])
    lstofham.append(hamiltonian)

print(lstofham)
for count, hamiltonian in enumerate(lstofham):
    hamiltonian_norm = 0.
    for term in hamiltonian.terms:
        hamiltonian_norm += abs(hamiltonian.terms[term])
    normalized_hamiltonian = deepcopy(hamiltonian)
    normalized_hamiltonian /= hamiltonian_norm


    eigenvalues, eigenvectors = get_eigenvalue_and_eigenvector(
        n_sites=num_qubits,
        hamiltonian=normalized_hamiltonian,
        k=2)
    print("Eigenvalues for {} are {}".format(coefs[count][0], eigenvalues))


    # Create a normalized equal superposition of the two eigenstates for numerical testing:
    initial_state_norm = 0.
    initial_state = [i+j for i,j in zip(eigenvectors[0], eigenvectors[1])]
    for amplitude in initial_state:
        initial_state_norm += abs(amplitude)**2
    normalized_initial_state = [amp / math.sqrt(initial_state_norm) for amp in initial_state]

    #initialize system qubits in this state:
    StatePreparation(normalized_initial_state) | system_qubits


    individual_terms = []
    initial_ancilla_wavefunction = []
    for term in normalized_hamiltonian.terms:
        coefficient = normalized_hamiltonian.terms[term]
        initial_ancilla_wavefunction.append(math.sqrt(abs(coefficient)))
        if coefficient < 0:
            individual_terms.append(QubitOperator(term, -1))
        else:
            individual_terms.append(QubitOperator(term))

    # Calculate the number of ancilla qubits required and pad
    # the ancilla wavefunction with zeros:
    num_ancilla_qubits = int(math.ceil(math.log(len(individual_terms), 2)))
    required_padding = 2**num_ancilla_qubits - len(initial_ancilla_wavefunction)
    initial_ancilla_wavefunction.extend([0]*required_padding)

    # Initialize ancillas by applying B
    ancillas = eng.allocate_qureg(num_ancilla_qubits)
    StatePreparation(initial_ancilla_wavefunction) | ancillas


    # Semiclassical iterative phase estimation
    bits_of_precision = 8
    pe_ancilla = eng.allocate_qubit()

    measurements = [0] * bits_of_precision

    for k in range(bits_of_precision):
        H | pe_ancilla
        with Control(eng, pe_ancilla):
            for i in range(2**(bits_of_precision-k-1)):
                W(eng=eng,
                  individual_terms=individual_terms,
                  initial_wavefunction=initial_ancilla_wavefunction,
                  ancilla_qubits=ancillas,
                  system_qubits=system_qubits)

        #inverse QFT using one qubit
        for i in range(k):
            if measurements[i]:
                R(-math.pi/(1 << (k - i))) | pe_ancilla

        H | pe_ancilla
        Measure | pe_ancilla
        eng.flush()
        measurements[k] = int(pe_ancilla)
        # put the ancilla in state |0> again
        if measurements[k]:
            X | pe_ancilla

    est_phase = sum(
        [(measurements[bits_of_precision - 1 - i]*1. / (1 << (i + 1)))
         for i in range(bits_of_precision)])

    print("For R = {}, we measured {} corresponding to energy {}".format(coefs[count][0], est_phase, math.cos(2*math.pi*est_phase)))

    #eng.backend.get_expectation_value(normalized_hamiltonian, system_qubits)





#!5 control bits:

# [-0.68615876 -0.08669864 -0.13392591 -0.1764226  -0.3275584  -0.43027061
#  -0.46427707 -0.4619599  -0.13392591 -0.1764226  -0.43027061 -0.4619599
#  -0.46427707 -0.4619599 ]
# We measured 0.263824462890625 corresponding to energy -0.0867524755202204

# with Dagger(eng):
#     StatePreparation(initial_ancilla_wavefunction) | ancillas
# measure_qb = eng.allocate_qubit()
# with Compute(eng):
#     All(X) | ancillas
# with Control(eng, ancillas):
#     X | measure_qb
# Uncompute(eng)
# eng.flush()
# eng.backend.get_probability('1', measure_qb)


# eng.backend.collapse_wavefunction(measure_qb, [1])
# eng.backend.get_expectation_value(normalized_hamiltonian, system_qubits)
