import os
import unittest
from typing import List


from min_part.molecules import h2_settings
from min_part.typing import GFROFragment
from min_part.utils import (
    open_frags,
    get_saved_file_names,
)


class InspectTest(unittest.TestCase):
    def test_h2_lb(self):
        global gfro_files

        config_settings = h2_settings
        global_id = str(19)
        parent_dir = f"../data/{config_settings.mol_name.lower()}"
        child_dir = os.path.join(parent_dir, "06-03", str(global_id))

        load = True
        if load:
            gfro_files, lr_files = get_saved_file_names(child_dir)

        for bond_length, gfro_file in zip(config_settings.xpoints, gfro_files):
            print(f"thetas and lambdas for {bond_length}")
            gfro_data: List[GFROFragment] = open_frags(gfro_file)
            for g_d in gfro_data:
                print(f"thetas: {g_d.thetas}")
                print(f"lambdas: {g_d.lambdas}")
