def mol_n2(i):
    return [["N", [0, 0, 0]], ["N", [0, 0, i]]]


def mol_h4(i):
    return [
        ["H", [0, 0, 0]],
        ["H", [0, 0, i]],
        ["H", [0, 0, 2 * i]],
        ["H", [0, 0, 3 * i]],
    ]


def mol_h2(i):
    return [["H", [0, 0, 0]], ["H", [0, 0, i]]]
