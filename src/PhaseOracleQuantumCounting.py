import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sin, asin, pi, sqrt, fabs, log2

from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.circuit.library import PhaseOracle
from qiskit.circuit.library import GroverOperator
from qiskit.circuit.library.basis_change.qft import QFT
from qiskit.visualization import plot_histogram

mpl.use('TkAgg')
plotting_enabled = False


def simulate_and_get_highest_measured_index(circuit: QuantumCircuit) -> int:
    aer_sim = Aer.get_backend('aer_simulator')
    transpiled_circuit = transpile(circuit, aer_sim)
    qobj = assemble(transpiled_circuit)
    job = aer_sim.run(qobj)
    histogram = job.result().get_counts()

    largest_measured_index_string = max(histogram, key=histogram.get)
    largest_measured_index = int(largest_measured_index_string, 2)

    if plotting_enabled:
        print(f"largest_measured_index: {largest_measured_index}")
        plot_histogram(histogram)
        plt.show()

    return largest_measured_index


def calculate_number_of_solutions_and_upper_error_bound(
        measured_index: int,
        number_of_phase_estimation_qubits: int,
        number_of_oracle_qubits: int
) -> (float, float):
    theta = (measured_index / (2 ** number_of_phase_estimation_qubits)) * pi * 2

    maximum_number_of_possible_solutions = 2 ** number_of_oracle_qubits
    number_of_solutions = maximum_number_of_possible_solutions * (sin(theta / 2) ** 2)

    maximum_error_bound = number_of_phase_estimation_qubits - 1
    upper_error_bound = (sqrt(2 * number_of_solutions * maximum_number_of_possible_solutions) +
                         maximum_number_of_possible_solutions / (2 ** (maximum_error_bound + 1))) * \
                        (2 ** (-maximum_error_bound))
    return number_of_solutions, upper_error_bound


def append_controlled_grover_iterations(
        circuit: QuantumCircuit,
        controlled_grover_circuit: QuantumCircuit,
        number_of_phase_estimation_qubits: int,
        total_number_of_qubits: int
) -> QuantumCircuit:
    current_number_of_iterations = 1
    for control_qubit in range(number_of_phase_estimation_qubits):
        for _ in range(current_number_of_iterations):
            circuit = circuit.compose(
                other=controlled_grover_circuit,
                qubits=[control_qubit] + [*range(number_of_phase_estimation_qubits, total_number_of_qubits)]
            )
        current_number_of_iterations *= 2
    return circuit


def quantum_counting_with_oracle(
        oracle: PhaseOracle,
        number_of_phase_estimation_qubits: int
) -> (float, float):
    number_of_oracle_qubits = oracle.num_qubits
    total_number_of_qubits = number_of_oracle_qubits + number_of_phase_estimation_qubits

    # Circuit with number_of_oracle_qubits + number_of_phase_estimation_qubits qubits
    # and number_of_phase_estimation_qubits classical bits
    qc = QuantumCircuit(
        total_number_of_qubits,
        number_of_phase_estimation_qubits
    )

    # Initialize all qubits to |+>
    for qubit in range(total_number_of_qubits):
        qc.h(qubit)

    # Controlled Grover iterations
    controlled_grover_circuit = GroverOperator(oracle=oracle, name="Grover").control()
    qc = append_controlled_grover_iterations(
        circuit=qc,
        controlled_grover_circuit=controlled_grover_circuit,
        number_of_phase_estimation_qubits=number_of_phase_estimation_qubits,
        total_number_of_qubits=total_number_of_qubits
    )

    # Do inverse QFT on counting qubits
    qc = qc.compose(
        other=QFT(num_qubits=number_of_phase_estimation_qubits, inverse=True),
        qubits=list(range(number_of_phase_estimation_qubits))
    )

    # Measure counting qubits
    qc.measure(
        qubit=range(number_of_phase_estimation_qubits),
        cbit=range(number_of_phase_estimation_qubits)
    )

    if plotting_enabled:
        qc.draw(output='mpl')
        plt.show()

    measured_index = simulate_and_get_highest_measured_index(circuit=qc)
    number_of_solutions, upper_error_bound = calculate_number_of_solutions_and_upper_error_bound(
        measured_index=measured_index,
        number_of_phase_estimation_qubits=number_of_phase_estimation_qubits,
        number_of_oracle_qubits=number_of_oracle_qubits
    )
    return number_of_solutions, upper_error_bound


def estimate_number_of_phase_estimation_qubits_given_expected_number_of_solutions(
        expected_number_of_solutions: int,
        tolerance: float,
        number_of_oracle_qubits: int
):
    maximum_number_of_possible_solutions = 2 ** number_of_oracle_qubits

    number_of_phase_estimation_qubits = 1
    expected_error = tolerance + 1
    while expected_error > tolerance:
        number_of_phase_estimation_qubits += 1
        expected_measured_index = int(round(2 ** (number_of_phase_estimation_qubits + 1) / (2 * pi) * \
                                asin(sqrt(expected_number_of_solutions / maximum_number_of_possible_solutions))))
        expected_measured_theta = (expected_measured_index / (2 ** number_of_phase_estimation_qubits)) * pi * 2
        expected_measured_number_of_solutions = maximum_number_of_possible_solutions * \
                                                (sin(expected_measured_theta / 2) ** 2)
        expected_error = fabs(expected_number_of_solutions - expected_measured_number_of_solutions)

    return number_of_phase_estimation_qubits


def estimate_number_of_phase_estimation_qubits(
        tolerance: float,
        number_of_oracle_qubits: int
):
    maximum_number_of_possible_solutions = 2 ** number_of_oracle_qubits
    number_of_phase_estimation_qubits = 2
    for expected_number_of_solutions in range(maximum_number_of_possible_solutions):
        number_of_phase_estimation_qubits_for_specific_output = estimate_number_of_phase_estimation_qubits_given_expected_number_of_solutions(
            expected_number_of_solutions=expected_number_of_solutions,
            tolerance=tolerance,
            number_of_oracle_qubits=number_of_oracle_qubits
        )
        number_of_phase_estimation_qubits = max(
            number_of_phase_estimation_qubits,
            number_of_phase_estimation_qubits_for_specific_output
        )
    return number_of_phase_estimation_qubits


def bit_string_to_disjunctive_normal_form(bit_string: str):
    assert isinstance(bit_string, str), "bit_string must be a string."
    assert len(bit_string) > 1, "bit_string must have at least 2 elements."
    assert log2(len(bit_string)).is_integer(), "length of bit_string must be a power of 2."
    assert all(c in '01' for c in bit_string), "bit_string can only contain the characters '0' or '1'."

    bit_string_length = len(bit_string)
    number_of_literals = int(log2(bit_string_length))

    disjunctive_normal_form_string = ''
    for bit_string_index in range(bit_string_length):
        # determine if the current value in the bit_string is true or false
        current_bool_value = bool(int(bit_string[bit_string_index]))

        if current_bool_value:
            # if it is true, we append a conjunction of literals
            current_conjunction_string = ''

            # check if we are not the first conjunction and if we are not, append an OR.
            if disjunctive_normal_form_string != '':
                current_conjunction_string += ' | '

            current_conjunction_string += '('
            # we iterate over all literals to construct the conjunction
            for literal_index in range(number_of_literals):

                # check if we are not the first literal in the conjunction and if we are not, append an AND.
                if literal_index > 0:
                    current_conjunction_string += ' & '

                # calculates whether the current literal is a 0 or 1
                # in the truth table for the current bit_string_index.
                literal_bool_value = bool(bit_string_index//(2**literal_index) % 2)
                if literal_bool_value:
                    current_conjunction_string += f'x{literal_index}'
                else:
                    current_conjunction_string += f'~x{literal_index}'
            current_conjunction_string += ')'

            disjunctive_normal_form_string += current_conjunction_string
        else:
            # dnf does not involve formulas for untrue values
            continue

    # if the whole bit string is 0s, we must produce a bit_string that is unsatisfiable
    if disjunctive_normal_form_string == '':
        disjunctive_normal_form_string = '~x0'
        for literal_index in range(number_of_literals):
            # check if we are not the first literal in the conjunction and if we are not, append an AND.
            disjunctive_normal_form_string += ' & '
            disjunctive_normal_form_string += f'x{literal_index}'

    return disjunctive_normal_form_string


def quantum_counting_with_bitstring(
        bit_string: str,
        number_of_phase_estimation_qubits: int
):
    disjunctive_normal_form = bit_string_to_disjunctive_normal_form(bit_string=bit_string)
    oracle = PhaseOracle(disjunctive_normal_form)
    number_of_solutions, upper_error_bound = quantum_counting_with_oracle(
        oracle=oracle,
        number_of_phase_estimation_qubits=number_of_phase_estimation_qubits
    )
    return number_of_solutions, upper_error_bound


def main():
    # bit_string = '1100000010001001'
    bit_string = '11000001'
    bit_string_length = len(bit_string)
    number_of_oracle_qubits = int(log2(bit_string_length))

    # the error bound has to be smaller than 0.5 to appropriately decide what the correct number of solutions are
    tolerance = 0.5
    number_of_phase_estimation_qubits = estimate_number_of_phase_estimation_qubits(
        tolerance=tolerance,
        number_of_oracle_qubits=number_of_oracle_qubits
    )
    # number_of_phase_estimation_qubits = 4

    print(f"number_of_oracle_qubits: {number_of_oracle_qubits}")
    print(f"number_of_phase_estimation_qubits: {number_of_phase_estimation_qubits}")

    number_of_solutions, upper_error_bound = quantum_counting_with_bitstring(
        bit_string=bit_string,
        number_of_phase_estimation_qubits=number_of_phase_estimation_qubits
    )
    print('%-14s | %-14s' % ('Measured #Sol', 'Error Bound'))
    print('%-14.1f | %-14.2f' % (number_of_solutions, upper_error_bound))
    print('')
    print('-' * 45)

    print('%-14s | %-14s | %-14s' % ('Expected #Sol', 'Measured #Sol', 'Error Bound'))
    print('-' * 45)
    for expected_number_of_solutions in range(2 ** number_of_oracle_qubits):
        bit_string = '1'*expected_number_of_solutions + '0'*(2**number_of_oracle_qubits - expected_number_of_solutions)
        number_of_solutions, upper_error_bound = quantum_counting_with_bitstring(
            bit_string=bit_string,
            number_of_phase_estimation_qubits=number_of_phase_estimation_qubits
        )
        print('%-14d | %-14.1f | %-14.2f' % (expected_number_of_solutions, number_of_solutions, upper_error_bound))


if __name__ == "__main__":
    main()

# sol_0 = 'x0 & ~x0 & x1 & x2 & x3'
# sol_1 = 'x0 & x1 & x2 & x3'
# sol_2 = sol_1 + ' | ~x0 & x1 & x2 & x3'
# sol_3 = sol_2 + ' | x0 & ~x1 & x2 & x3'
# sol_4 = sol_3 + ' | ~x0 & ~x1 & x2 & x3'
# sol_5 = sol_4 + ' | x0 & x1 & ~x2 & x3'
# sol_6 = sol_5 + ' | ~x0 & x1 & ~x2 & x3'
# sol_7 = sol_6 + ' | x0 & ~x1 & ~x2 & x3'
# sol_8 = sol_7 + ' | ~x0 & ~x1 & ~x2 & x3'
# sol_9 = sol_8 + ' | x0 & x1 & x2 & ~x3'
# sol_10 = sol_9 + ' | ~x0 & x1 & x2 & ~x3'
# sol_11 = sol_10 + ' | x0 & ~x1 & x2 & ~x3'
# sol_12 = sol_11 + ' | ~x0 & ~x1 & x2 & ~x3'
# sol_13 = sol_12 + ' | x0 & x1 & ~x2 & ~x3'
# sol_14 = sol_13 + ' | ~x0 & x1 & ~x2 & ~x3'
# sol_15 = sol_14 + ' | x0 & ~x1 & ~x2 & ~x3'
#
# solutions = [sol_0,
#              sol_1,
#              sol_2,
#              sol_3,
#              sol_4,
#              sol_5,
#              sol_6,
#              sol_7,
#              sol_8,
#              sol_9,
#              sol_10,
#              sol_11,
#              sol_12,
#              sol_13,
#              sol_14,
#              sol_15]
