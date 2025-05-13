import scipy as sp


def sum_frags(frags):
    min_e_each_frag = []
    for i, frag in enumerate(frags):
        values, wectors = sp.linalg.eigh(frag.toarray())
        min_e_each_frag.append(min(values))
    return sum(min_e_each_frag)
