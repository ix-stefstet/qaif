import numpy as np

from src.py.quantum_counting.quantum_counting import QuantumCounting


def main():
    s_above = 6
    s_below = 0

    while abs(s_above - s_below) > 0.01:
        print('-' * 45)

        s_threshold = (s_above + s_below) / 2
        print(f"s_above: {s_above} , s_threshold: {s_threshold}, s_below: {s_below}")

        bit_string = calculate_truth_table_string(s_threshold)
        number_of_oracle_qubits = int(np.log2(len(bit_string)))

        quantum_counting = QuantumCounting(
            number_of_oracle_qubits=number_of_oracle_qubits
        )
        number_of_solutions, upper_error_bound = quantum_counting.quantum_counting_with_bitstring(
            bit_string=bit_string,
        )

        print('%-14s | %-14s' % ('Measured #Sol', 'Error Bound'))
        print('%-14.1f | %-14.2f' % (number_of_solutions, upper_error_bound))

        if number_of_solutions == 0:
            s_above = s_threshold
        else:
            s_below = s_threshold


def calculate_truth_table_string(s_threshold):
    r_1 = 1  # return deviation asset 1
    r_2 = 2  # return deviation asset 2

    sigma_1 = 3  # standard deviation asset 1
    sigma_2 = 2  # standard deviation asset 2

    cor = 0.5  # Correlation between asset 1 and 2, lies between 0 and 1

    s = ''
    for i in range(0, 4):
        s += is_sharpe_ratio_above_threshold(i, r_1, r_2, sigma_1, sigma_2, cor, s_threshold)

    return s  # Truth table string


def is_sharpe_ratio_above_threshold(allocation_in_asset1, r_1, r_2, sigma_1, sigma_2, cor, threshold):
    w_1 = allocation_in_asset1 / 3  # Allocation in Asset 1
    w_2 = 1 - w_1  # Allocation in Asset 1

    r = w_1 * r_1 + w_2 * r_2  # Portfolio return

    # Portfolio standard deviation
    std = (w_1 ** 2 * sigma_1 ** 2 + w_2 ** 2 * sigma_2 ** 2 + 2 * cor * w_1 * w_2 * sigma_1 * sigma_2) ** 0.5

    sharpe = r / std  # Sharpe ratio

    if sharpe > threshold:
        return '1'
    else:
        return '0'


if __name__ == '__main__':
    main()
