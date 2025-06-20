from hypothesis import strategies as st

from min_part.gfro_decomp import gfro_decomp
from min_part.lr_decomp import lr_decomp
from min_part.testing_utils.sim_tensor import make_tensors_h2


def specfic_gfro_decomp(bond_length):
    H_const, H_obt, H_tbt = make_tensors_h2(bond_length)
    return H_obt, H_tbt, gfro_decomp(H_tbt), None


@st.composite
def H_2_LR(draw):
    bond_length = draw(st.floats(0.2, 3))
    H_const, H_obt, H_tbt = make_tensors_h2(bond_length)
    lr_h2_frags = lr_decomp(H_tbt)
    print(f"bond length: {bond_length}")
    return H_obt, H_tbt, lr_h2_frags, bond_length


@st.composite
def H_2_GFRO(draw):
    bond_length = draw(st.floats(0.2, 3))
    H_const, H_obt, H_tbt = make_tensors_h2(bond_length)
    gfro_h2_frags = gfro_decomp(H_tbt)
    print(f"bond length: {bond_length}")
    return H_obt, H_tbt, gfro_h2_frags, bond_length
