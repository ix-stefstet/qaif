import numpy as np

from src.py.quantum_counting.quantum_counting import QuantumCounting


def main():
    # bit_string = '1100000010001001'
    bit_string = '11000001'
    number_of_oracle_qubits = int(np.log2(len(bit_string)))

    quantum_counting = QuantumCounting(
        number_of_oracle_qubits=number_of_oracle_qubits
    )
    number_of_solutions, upper_error_bound = quantum_counting.quantum_counting_with_bitstring(
        bit_string=bit_string,
    )
    print('%-14s | %-14s' % ('Measured #Sol', 'Error Bound'))
    print('%-14.1f | %-14.2f' % (number_of_solutions, upper_error_bound))
    print('')
    print('-' * 45)

    print('%-14s | %-14s | %-14s' % ('Expected #Sol', 'Measured #Sol', 'Error Bound'))
    print('-' * 45)
    for expected_number_of_solutions in range(2 ** number_of_oracle_qubits):
        bit_string = '1'*expected_number_of_solutions + '0'*(2**number_of_oracle_qubits - expected_number_of_solutions)
        number_of_solutions, upper_error_bound = quantum_counting.quantum_counting_with_bitstring(
            bit_string=bit_string,
        )
        print('%-14d | %-14.1f | %-14.2f' % (expected_number_of_solutions, number_of_solutions, upper_error_bound))


if __name__ == "__main__":
    main()
