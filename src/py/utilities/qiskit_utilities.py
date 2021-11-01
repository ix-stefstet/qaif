import numpy as np

from qiskit import QuantumCircuit, transpile, Aer
from qiskit.circuit.library import PhaseOracle
from qiskit.circuit.library import GroverOperator


def simulate_quantum_circuit(
        quantum_circuit: QuantumCircuit
) -> dict[str, int]:
    aer_sim = Aer.get_backend('aer_simulator')
    transpiled_circuit = transpile(quantum_circuit, aer_sim)
    job = aer_sim.run(transpiled_circuit)
    histogram = job.result().get_counts()

    return histogram


def get_highest_measured_index_from_histogram(
        histogram: dict[str, int]
) -> int:
    largest_measured_index_string = max(histogram, key=histogram.get)
    largest_measured_index = int(largest_measured_index_string, 2)

    return largest_measured_index


def append_2x_controlled_grover_iterations_to_quantum_circuit(
        oracle: PhaseOracle,
        quantum_circuit: QuantumCircuit,
        number_of_phase_estimation_qubits: int,
        total_number_of_qubits: int
) -> QuantumCircuit:
    controlled_grover_circuit = GroverOperator(oracle=oracle, name="Grover").control()

    current_number_of_iterations = 1
    for control_qubit in range(number_of_phase_estimation_qubits):
        for _ in range(current_number_of_iterations):
            quantum_circuit = quantum_circuit.compose(
                other=controlled_grover_circuit,
                qubits=[control_qubit] + [*range(number_of_phase_estimation_qubits, total_number_of_qubits)]
            )
        current_number_of_iterations *= 2
    return quantum_circuit


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
        expected_measured_index = int(
            round(
                2 ** (number_of_phase_estimation_qubits + 1) / (2 * np.pi) *
                np.arcsin(
                    np.sqrt(
                        expected_number_of_solutions / maximum_number_of_possible_solutions
                    )
                )
            )
        )
        expected_measured_theta = (expected_measured_index / (2 ** number_of_phase_estimation_qubits)) * np.pi * 2
        expected_measured_number_of_solutions = maximum_number_of_possible_solutions * np.sin(
            expected_measured_theta / 2
        ) ** 2
        expected_error = np.fabs(expected_number_of_solutions - expected_measured_number_of_solutions)

    return number_of_phase_estimation_qubits


def estimate_number_of_phase_estimation_qubits(
        tolerance: float,
        number_of_oracle_qubits: int
):
    maximum_number_of_possible_solutions = 2 ** number_of_oracle_qubits
    number_of_phase_estimation_qubits = 2
    for expected_number_of_solutions in range(maximum_number_of_possible_solutions):
        number_of_phase_estimation_qubits_for_specific_output = \
            estimate_number_of_phase_estimation_qubits_given_expected_number_of_solutions(
                expected_number_of_solutions=expected_number_of_solutions,
                tolerance=tolerance,
                number_of_oracle_qubits=number_of_oracle_qubits
            )
        number_of_phase_estimation_qubits = max(
            number_of_phase_estimation_qubits,
            number_of_phase_estimation_qubits_for_specific_output
        )
    return number_of_phase_estimation_qubits
