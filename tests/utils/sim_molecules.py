import os.path

from hypothesis import strategies as st
from openfermion import eigenspectrum

from min_part.gfro_decomp import gfro_decomp
from min_part.lr_decomp import lr_decomp
from min_part.molecules import h4_settings
from min_part.tensor import obt2op
from min_part.utils import save_frags, open_frags
from tests.utils.sim_tensor import make_tensors_h2, get_tensors


def specific_gfro_decomp(bond_length, tol=1e-6):
    H_const, H_obt, H_tbt = make_tensors_h2(bond_length)
    frag_folder = (
        "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.frags/gfro"
    )
    frag_path = os.path.join(frag_folder, str(bond_length))
    if os.path.exists(f"{frag_path}.pkl"):
        gfro_h2_frags = open_frags(frag_path)
        print("used saved frags")
    else:
        gfro_h2_frags = gfro_decomp(H_tbt, tol)
        save_frags(gfro_h2_frags, file_name=frag_path)
    return H_obt, H_tbt, gfro_h2_frags, bond_length


def specific_lr_decomp(bond_length):
    H_const, H_obt, H_tbt = make_tensors_h2(bond_length)
    frag_folder = "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.frags/lr"
    frag_path = os.path.join(frag_folder, str(bond_length))
    if os.path.exists(f"{frag_path}.pkl"):
        lr_frags = open_frags(frag_path)
        print("used saved frags")
    else:
        lr_frags = lr_decomp(H_tbt)
        save_frags(lr_frags, file_name=frag_path)
    return H_obt, H_tbt, lr_frags, bond_length


@st.composite
def H_4_LR(draw):
    bond_length = draw(st.floats(0.2, 3))
    H_const, H_obt, H_tbt = get_tensors(h4_settings, bond_length)
    frag_path = os.path.join(h4_settings.frag_folder, str(bond_length))
    if os.path.exists(f"{frag_path}.pkl"):
        lr_frags = open_frags(frag_path)
    else:
        lr_frags = lr_decomp(H_tbt)
        save_frags(lr_frags, file_name=frag_path)
    print(f"bond length: {bond_length}")
    print(min(eigenspectrum(H_const + obt2op(H_obt))))
    return H_obt, H_tbt, lr_frags, bond_length


@st.composite
def H_2_LR(draw):
    bond_length = draw(st.floats(0.2, 3))
    H_const, H_obt, H_tbt = make_tensors_h2(bond_length)
    frag_folder = "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.frags/lr"
    frag_path = os.path.join(frag_folder, str(bond_length))
    if os.path.exists(f"{frag_path}.pkl"):
        lr_frags = open_frags(frag_path)
        print("used saved frags")
    else:
        lr_frags = lr_decomp(H_tbt)
        save_frags(lr_frags, file_name=frag_path)
    print(f"bond length: {bond_length}")
    return H_obt, H_tbt, lr_frags, bond_length


@st.composite
def H_2_GFRO(draw, tol=1e-6):
    bond_length = draw(st.floats(0.2, 3))
    H_const, H_obt, H_tbt = make_tensors_h2(bond_length)
    frag_folder = (
        "/Users/lucyhao/Obsidian 10.41.25/GradSchool/Code/qc/tests/.frags/gfro"
    )
    frag_path = os.path.join(frag_folder, str(bond_length))
    if os.path.exists(f"{frag_path}.pkl"):
        gfro_h2_frags = open_frags(frag_path)
        print("used saved frags")
    else:
        gfro_h2_frags = gfro_decomp(H_tbt, tol)
        save_frags(gfro_h2_frags, file_name=frag_path)
    print(f"bond length: {bond_length}")
    return H_obt, H_tbt, gfro_h2_frags, bond_length
