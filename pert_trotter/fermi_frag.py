from openfermion import (
    get_sparse_operator,
    FermionOperator,
)
from opt_einsum import contract

from . import config
import pickle
import numpy as np

from .ffrag_utils import (
    LR_frags_generator,
    gfro_frags_generator,
    FRO_frags_generator,
    get_u_from_angles,
    get_coeff_mat_from_coeffs,
    sdgfro_frags_generator,
)
from .tensor_utils import get_chem_tensors, obt2tbt, obt2op, spac2spin, tbt2op


# projector_func used in all functions is the function that projects the sparse arrays on to the correct fermionic symmtery subspace


# Do LR
def Do_LR(H_FO, shrink_frag, CISD, save=True, projector_func=None):
    if CISD == False:
        excitations = None
    else:
        excitations = 2

    const, obt, tbt = get_chem_tensors(H_FO)
    obt_op = obt2op(obt)

    # Obtaining LR fragments as list of FermionOperators and (coeffs, angles) defining the fragments.
    lowrank_fragments, lowrank_params = LR_frags_generator(
        tbt, tol=1e-5, ret_params=True
    )

    # Filtering out small fragments
    LR_fragments = []
    LR_params = []
    for i in range(len(lowrank_params)):
        frag = lowrank_fragments[i]
        if frag.induced_norm(2) > 1e-6:
            LR_fragments.append(frag)
            LR_params.append(lowrank_params[i])

    all_frag_ops = [const * FermionOperator.identity(), obt_op]
    all_frag_ops += LR_fragments

    if save:
        savefilename = config.savepath + "lr/" + config.mol_name + "_lr_params.pkl"
        savefile = {"fragment_ops": LR_fragments, "params": LR_params}
        with open(savefilename, "wb") as out_file:
            pickle.dump(savefile, out_file)

    if not shrink_frag:
        sparse_fragments_list = [
            get_sparse_operator(frag, n_qubits=config.n_qubits) for frag in all_frag_ops
        ]
    else:
        sparse_fragments_list = [
            projector_func(frag, excitation_level=excitations) for frag in all_frag_ops
        ]

    return np.array(sparse_fragments_list)


# Do SD LR
def Do_SD_LR(H_FO, shrink_frag, CISD, save=True, projector_func=None):
    if CISD == False:
        excitations = None
    else:
        excitations = 2

    const, obt, tbt = get_chem_tensors(H_FO)
    tbt_new = obt2tbt(obt)
    tbt += tbt_new

    # Obtaining LR fragments as list of FermionOperators and (coeffs, angles) defining the fragments.
    lowrank_fragments, lowrank_params = LR_frags_generator(
        tbt, tol=1e-5, ret_params=True
    )

    # Filtering out small fragments
    LR_fragments = []
    LR_params = []
    for i in range(len(lowrank_params)):
        frag = lowrank_fragments[i]
        if frag.induced_norm(2) > 1e-6:
            LR_fragments.append(frag)
            LR_params.append(lowrank_params[i])

    all_frag_ops = [const * FermionOperator.identity()]
    all_frag_ops += LR_fragments

    if save == True:
        savefilename = config.savepath + "sdlr/" + config.mol_name + "_sdlr_params.pkl"
        savefile = {"fragment_ops": LR_fragments, "params": LR_params}
        with open(savefilename, "wb") as out_file:
            pickle.dump(savefile, out_file)

    if shrink_frag == False:
        sparse_fragments_list = [
            get_sparse_operator(frag, n_qubits=config.n_qubits) for frag in all_frag_ops
        ]
    else:
        sparse_fragments_list = [
            projector_func(frag, excitation_level=excitations) for frag in all_frag_ops
        ]

    return np.array(sparse_fragments_list)


# Do LR LCU
def Do_LR_LCU(H_FO, shrink_frag, CISD, save=True, load=True, projector_func=None):
    if CISD == False:
        excitations = None
    else:
        excitations = 2

    const, obt, tbt = get_chem_tensors(H_FO)
    obt_op = obt2op(obt)

    if load == False:
        print("Load parameter has been set to False, so LR will be done first.")
        # Obtaining LR fragments as list of FermionOperators and (coeffs, angles) defining the fragments.
        LR_fragments_org, LR_params_org = LR_frags_generator(
            tbt, tol=1e-5, ret_params=True
        )

        # Filtering out small fragments
        LR_fragments = []
        LR_params = []
        for i in range(len(LR_params_org)):
            frag = LR_fragments_org[i]
            if frag.induced_norm(2) > 1e-6:
                LR_fragments.append(frag)
                LR_params.append(LR_params_org[i])
    else:
        with open(
            config.savepath + "lr/" + config.mol_name + "_lr_params.pkl", "rb"
        ) as in_file:
            loadfile = pickle.load(in_file)
        LR_fragments = loadfile["fragment_ops"]
        LR_params = loadfile["params"]

    # Rewriting LR fragments as reflections.
    LR_LCU_frag_ops = []
    obt_correction = FermionOperator.zero()
    const_correction = 0
    for temp_params in LR_params:
        coeff_mat = temp_params[0]

        term_1_spac = contract(
            "ij,pi,qi,rj,sj->pqrs",
            coeff_mat,
            temp_params[1],
            temp_params[1],
            temp_params[1],
            temp_params[1],
        )
        term_1_spin = spac2spin(term_1_spac)
        term_1_op = tbt2op(term_1_spin)

        term_2_spac = contract(
            "ij,pi,qi->pq", coeff_mat / 2, temp_params[1], temp_params[1]
        )
        term_2_spin = 2 * spac2spin(
            term_2_spac
        )  # Factor of 2 appears due to the spin degree of freedom.
        term_2_op = obt2op(term_2_spin)

        term_3_spin = (
            4 * np.sum(coeff_mat) / 4
        )  # Factor of 4 appears due to the spin degree of freedom. We are assuming LR is done in spacial orbital basis.

        new_frag = term_1_op - 2 * term_2_op + term_3_spin * FermionOperator.identity()

        const_correction -= term_3_spin
        obt_correction += 2 * term_2_op
        LR_LCU_frag_ops.append(new_frag)

    const += const_correction
    obt_op += obt_correction
    all_frag_ops = [const * FermionOperator.identity()] + [obt_op] + LR_LCU_frag_ops

    if save == True:
        savefilename = config.savepath + "lrlcu/" + config.mol_name + "_lrlcu_ops.pkl"
        with open(savefilename, "wb") as out_file:
            pickle.dump(all_frag_ops, out_file)

    if shrink_frag == False:
        sparse_fragments_list = [
            get_sparse_operator(frag, n_qubits=config.n_qubits) for frag in all_frag_ops
        ]
    else:
        sparse_fragments_list = [
            projector_func(frag, excitation_level=excitations) for frag in all_frag_ops
        ]

    return np.array(sparse_fragments_list)


# Do GFRO
def Do_GFRO(
    H_FO, shrink_frag, CISD, tol=1e-6, save=True, spacial=False, projector_func=None
):
    if CISD == False:
        excitations = None
    else:
        excitations = 2

    const, obt, tbt = get_chem_tensors(H_FO)
    obt_op = obt2op(obt)

    gfro_fragments, gfro_params = gfro_frags_generator(
        tbt, ret_params=True, tol=tol, spacial=spacial
    )
    all_frag_ops = [const * FermionOperator.identity(), obt_op]
    all_frag_ops += gfro_fragments

    if save == True:
        savefilename = config.savepath + "gfro/" + config.mol_name + "_gfro_params.pkl"
        savefile = {"fragment_ops": all_frag_ops, "params": gfro_params}
        with open(savefilename, "wb") as out_file:
            pickle.dump(savefile, out_file)

    if shrink_frag == False:
        sparse_fragments_list = [
            get_sparse_operator(frag, n_qubits=config.n_qubits) for frag in all_frag_ops
        ]
    else:
        sparse_fragments_list = [
            projector_func(frag, excitation_level=excitations) for frag in all_frag_ops
        ]

    return np.array(sparse_fragments_list)


# Do SD GFRO (new)
def Do_SD_GFRO(
    H_FO, shrink_frag, CISD, tol=1e-6, save=True, spacial=False, projector_func=None
):
    if CISD == False:
        excitations = None
    else:
        excitations = 2

    const, obt, tbt = get_chem_tensors(H_FO)

    gfro_fragments, gfro_params_1, gfro_params_2 = sdgfro_frags_generator(
        obt, tbt, tol=tol, ret_params=True, spacial=False
    )
    all_frag_ops = [const * FermionOperator.identity()] + gfro_fragments

    if save == True:
        savefilename = (
            config.savepath + "sdgfro/" + config.mol_name + "_sdgfro_params.pkl"
        )
        savefile = {
            "fragment_ops": all_frag_ops,
            "params_1": gfro_params_1,
            "params_2": gfro_params_2,
        }
        with open(savefilename, "wb") as out_file:
            pickle.dump(savefile, out_file)

    if shrink_frag == False:
        sparse_fragmetns_list = [
            get_sparse_operator(frag, n_qubits=config.n_qubits) for frag in all_frag_ops
        ]
    else:
        sparse_fragmetns_list = [
            projector_func(frag, excitation_level=excitations) for frag in all_frag_ops
        ]

    return np.array(sparse_fragmetns_list)


# Do GFRO LCU
def Do_GFRO_LCU(
    H_FO,
    shrink_frag,
    CISD,
    tol=1e-6,
    save=True,
    load=True,
    spacial=False,
    projector_func=None,
):
    if CISD == False:
        excitations = None
    else:
        excitations = 2

    const, obt, tbt = get_chem_tensors(H_FO)
    obt_op = obt2op(obt)
    N = int(config.n_qubits / 2)

    if load == False:
        print("Load parameter has been set to False, so GFRO will be done first.")
        gfro_fragments, gfro_params = gfro_frags_generator(
            tbt, ret_params=True, tol=tol, spacial=spacial
        )
    else:
        with open(
            config.savepath + "gfro/" + config.mol_name + "_gfro_params.pkl", "rb"
        ) as in_file:
            loadfile = pickle.load(in_file)
        gfro_fragments = loadfile["fragment_ops"]
        gfro_params = loadfile["params"]

    GFRO_LCU_frag_ops = []
    obt_correction = FermionOperator.zero()
    const_correction = 0

    frag_num = 1
    for temp_params, frag in zip(gfro_params, gfro_fragments):
        coeff_mat = get_coeff_mat_from_coeffs(temp_params[0], N)
        u = get_u_from_angles(temp_params[1], N)

        term_1_spac = contract("ij,pi,qi,rj,sj->pqrs", coeff_mat, u, u, u, u)
        term_1_spin = spac2spin(term_1_spac)
        term_1_op = tbt2op(term_1_spin)

        term_2_spac = contract("ij,pi,qi->pq", coeff_mat / 2, u, u)
        term_2_spin = 2 * spac2spin(
            term_2_spac
        )  # Factor of 2 appears due to the spin degree of freedom.
        term_2_op = obt2op(term_2_spin)

        term_3_spin = (
            4 * np.sum(coeff_mat) / 4
        )  # Factor of 4 appears due to introducing spin degree of freedom. We are assuming GFRO is done in spacial orbital basis.

        new_frag = term_1_op - 2 * term_2_op + term_3_spin * FermionOperator.identity()

        const_correction -= term_3_spin
        obt_correction += 2 * term_2_op
        GFRO_LCU_frag_ops.append(new_frag)

        print(frag_num)
        frag_num += 1

    const += const_correction
    obt_op += obt_correction
    all_frag_ops = [const * FermionOperator.identity()] + [obt_op] + GFRO_LCU_frag_ops

    if save == True:
        savefilename = (
            config.savepath + "gfrolcu/" + config.mol_name + "_gfrolcu_frag_ops.pkl"
        )
        with open(savefilename, "wb") as out_file:
            pickle.dump(all_frag_ops, out_file)

    if shrink_frag == False:
        Tot_GFRO_LCU_frag_sparse = [
            get_sparse_operator(frag, n_qubits=config.n_qubits) for frag in all_frag_ops
        ]
    else:
        Tot_GFRO_LCU_frag_sparse = [
            projector_func(frag, excitation_level=excitations) for frag in all_frag_ops
        ]

    return np.array(Tot_GFRO_LCU_frag_sparse)


# Do FRO
def Do_FRO(H_FO, N_frags, shrink_frag, CISD, save=True, projector_func=None):
    if CISD == False:
        excitations = None
    else:
        excitations = 2

    const, obt, tbt = get_chem_tensors(H_FO)
    obt_op = obt2op(obt)

    FRO_fragments = FRO_frags_generator(tbt, N_frags=N_frags)
    all_frag_ops = [const * FermionOperator.identity()] + [obt_op] + FRO_fragments

    if save == True:
        savefilename = config.savepath + "fro/" + config.mol_name + "_fro_frags.pkl"
        with open(savefilename, "wb") as out_file:
            pickle.dump(all_frag_ops, out_file)

    if shrink_frag == False:
        sparse_fragments_list = [
            get_sparse_operator(frag, n_qubits=config.n_qubits) for frag in all_frag_ops
        ]
    else:
        sparse_fragments_list = [
            projector_func(frag, excitation_level=excitations) for frag in all_frag_ops
        ]

    return np.array(sparse_fragments_list)


def Do_Fermi_Partitioning(
    H_FO,
    type: str,
    shrink_frag=True,
    CISD=False,
    tol=1e-4,
    save=True,
    load=True,
    spacial=False,
    N_frags=20,
    projector_func=None,
):
    if type.upper().replace(" ", "_") == "LR":
        return Do_LR(
            H_FO,
            shrink_frag=shrink_frag,
            CISD=CISD,
            save=save,
            projector_func=projector_func,
        )
    elif type.upper().replace(" ", "_") == "SD_LR" or (type.upper() == "SDLR"):
        return Do_SD_LR(
            H_FO,
            shrink_frag=shrink_frag,
            CISD=CISD,
            save=save,
            projector_func=projector_func,
        )
    elif type.upper().replace(" ", "_") == "LR_LCU" or (type.upper() == "LRLCU"):
        return Do_LR_LCU(
            H_FO,
            shrink_frag=shrink_frag,
            CISD=CISD,
            save=save,
            load=load,
            projector_func=projector_func,
        )
    elif type.upper().replace(" ", "_") == "GFRO":
        return Do_GFRO(
            H_FO,
            tol=tol,
            shrink_frag=shrink_frag,
            CISD=CISD,
            save=save,
            spacial=spacial,
            projector_func=projector_func,
        )
    elif (type.upper().replace(" ", "_") == "SD_GFRO") or (type.upper() == "SDGFRO"):
        return Do_SD_GFRO(
            H_FO,
            tol=tol,
            shrink_frag=shrink_frag,
            CISD=CISD,
            save=save,
            spacial=spacial,
            projector_func=projector_func,
        )
    elif type.upper().replace(" ", "_") == "GFRO_LCU" or (type.upper() == "GFROLCU"):
        return Do_GFRO_LCU(
            H_FO,
            tol=tol,
            shrink_frag=shrink_frag,
            CISD=CISD,
            save=save,
            load=load,
            spacial=spacial,
            projector_func=projector_func,
        )
    elif (
        type.upper().replace(" ", "_") == "FRO"
    ):  # Requires a predefined global parameter, N_frags
        return Do_FRO(
            H_FO,
            N_frags,
            shrink_frag=shrink_frag,
            CISD=CISD,
            save=save,
            projector_func=projector_func,
        )
