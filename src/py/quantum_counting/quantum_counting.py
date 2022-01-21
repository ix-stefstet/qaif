import numpy as np
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracle
from qiskit.circuit.library.basis_change.qft import QFT

from src.py.utilities.qiskit_utilities import \
    simulate_quantum_circuit, \
    get_highest_measured_index_from_histogram, \
    append_2x_controlled_grover_iterations_to_quantum_circuit, \
    estimate_number_of_phase_estimation_qubits
from src.py.utilities.converters import bit_string_to_disjunctive_normal_form


class QuantumCounting:
    def __init__(
        self,
        number_of_oracle_qubits: int,
        phase_estimation_tolerance: Optional[float] = 0.5,
        number_of_phase_estimation_qubits: Optional[int] = None
    ):
        assert phase_estimation_tolerance is not None or number_of_phase_estimation_qubits is not None, \
            'either phase_estimation_tolerance has to be specified to calculate the optimal number of ' \
            'phase estimation qubits or number_of_phase_estimation_qubits has to be specified directly.'

        self.number_of_oracle_qubits = number_of_oracle_qubits
        if number_of_phase_estimation_qubits is not None:
            self.number_of_phase_estimation_qubits = number_of_phase_estimation_qubits
        else:
            self.number_of_phase_estimation_qubits = estimate_number_of_phase_estimation_qubits(
                tolerance=phase_estimation_tolerance,
                number_of_oracle_qubits=self.number_of_oracle_qubits
            )
        self.total_number_of_qubits = self.number_of_oracle_qubits + self.number_of_phase_estimation_qubits

    def __calculate_number_of_solutions_and_upper_error_bound(
            self,
            highest_measured_index: int
    ) -> (float, float):
        theta = (highest_measured_index / (2 ** self.number_of_phase_estimation_qubits)) * np.pi * 2

        maximum_number_of_possible_solutions = 2 ** self.number_of_oracle_qubits
        number_of_solutions = maximum_number_of_possible_solutions * (np.sin(theta / 2) ** 2)

        maximum_error_bound = self.number_of_phase_estimation_qubits - 1
        upper_error_bound = (np.sqrt(2 * number_of_solutions * maximum_number_of_possible_solutions) +
                             maximum_number_of_possible_solutions / (2 ** (maximum_error_bound + 1))) * \
                            (2 ** (-maximum_error_bound))
        return number_of_solutions, upper_error_bound

    def quantum_counting_with_oracle(
            self,
            oracle: PhaseOracle
    ) -> (float, float):
        assert oracle.num_qubits == self.number_of_oracle_qubits, \
            "number_of_oracle_qubits must match the number of qubits of the oracle"

        # Circuit with number_of_oracle_qubits + number_of_phase_estimation_qubits qubits
        # and number_of_phase_estimation_qubits classical bits
        qc = QuantumCircuit(
            self.total_number_of_qubits,
            self.number_of_phase_estimation_qubits
        )

        # Initialize all qubits to |+>
        for qubit in range(self.total_number_of_qubits):
            qc.h(qubit)

        qc = append_2x_controlled_grover_iterations_to_quantum_circuit(
            oracle=oracle,
            quantum_circuit=qc,
            number_of_phase_estimation_qubits=self.number_of_phase_estimation_qubits,
            total_number_of_qubits=self.total_number_of_qubits
        )

        # Do inverse QFT on counting qubits
        qc = qc.compose(
            other=QFT(num_qubits=self.number_of_phase_estimation_qubits, inverse=True),
            qubits=list(range(self.number_of_phase_estimation_qubits))
        )

        # Measure counting qubits
        qc.measure(
            qubit=range(self.number_of_phase_estimation_qubits),
            cbit=range(self.number_of_phase_estimation_qubits)
        )

        histogram = simulate_quantum_circuit(
            quantum_circuit=qc
        )
        highest_measured_index = get_highest_measured_index_from_histogram(
            histogram=histogram
        )

        number_of_solutions, upper_error_bound = self.__calculate_number_of_solutions_and_upper_error_bound(
            highest_measured_index=highest_measured_index
        )
        return number_of_solutions, upper_error_bound

    def quantum_counting_with_bitstring(
            self,
            bit_string: str
    ) -> (float, float):
        disjunctive_normal_form = bit_string_to_disjunctive_normal_form(bit_string=bit_string)
        oracle = PhaseOracle(disjunctive_normal_form)
        number_of_solutions, upper_error_bound = self.quantum_counting_with_oracle(
            oracle=oracle,
        )
        return number_of_solutions, upper_error_bound
