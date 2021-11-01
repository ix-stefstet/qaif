import numpy as np


def bit_string_to_disjunctive_normal_form(bit_string: str):
    assert isinstance(bit_string, str), "bit_string must be a string."
    assert len(bit_string) > 1, "bit_string must have at least 2 elements."
    assert np.log2(len(bit_string)).is_integer(), "length of bit_string must be a power of 2."
    assert all(c in '01' for c in bit_string), "bit_string can only contain the characters '0' or '1'."

    bit_string_length = len(bit_string)
    number_of_literals = int(np.log2(bit_string_length))

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
                literal_bool_value = bool(bit_string_index // (2 ** literal_index) % 2)
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
