import sys
import os
import shutil
import uuid
import subprocess as sp
from multiprocessing import Pool
import numpy as np

from chemlab.util.file_system import ELEMENT_DICT,qchem_out_force,qchem_file
from chemlab.scripts.base import QchemBaseScript
from chemlab.config.config_loader import ConfigBase

FIELD_REGISTRY = {
    "energy": {
        "save_name": "energy",
        "loader": lambda ctx: np.fromfile(ctx["cache_path"] + "99.0", dtype="f8", count=2)[1],
    },
    "qm_grad": {
        "save_name": "qm_grad",
        "loader": lambda ctx: np.fromfile(ctx["cache_path"] + "131.0", dtype="f8").reshape(-1, 3),
    },
    "qm_coord": {
        "save_name": "qm_coord",
        "loader": lambda ctx: ctx["qmout"].molecule.xyz,
    },
    "mm_coord": {
        "save_name": "mm_coord",
        "loader": lambda ctx: ctx["qmout"].external_charges.mm_pos,
    },
    "mm_charge": {
        "save_name": "mm_charge",
        "loader": lambda ctx: ctx["qmout"].external_charges.mm_charge,
    },
    "mm_esp": {
        "save_name": "mm_esp",
        "loader": lambda ctx: np.loadtxt(ctx["esp_file"], dtype=float, skiprows=3),
        "merge_post": lambda arr: arr[:, :, -1],
    },
    "mm_esp_grad": {
        "save_name": "mm_esp_grad",
        "loader": lambda ctx: -np.loadtxt(ctx["efld_file"], max_rows=ctx["n_mm"], dtype=float),
    },
}

class QMMMTrainSetDataConfig(ConfigBase):
    section_name = "qmmm_training_set_data"


class QMMMTrainSetData(QchemBaseScript):
    name = "example_script"
    config = QMMMTrainSetDataConfig

    def run(self, cfg):
        if cfg.method == "gas":
            self.run_gas(cfg)
            return 0
        elif cfg.method == "qmmm":
            self.run_qmmm(cfg)
            return 0
        else:
            return KeyError(f"Method {cfg.method} Not Supported")
    def run_qmmm(self,cfg):
        qmmmpath = cfg.qmmmpath
        cache_path = cfg.cache_path
        outpath = cfg.outpath
        prefix = cfg.prefix
        windows = cfg.windows
        nframes = cfg.nframes
        os.makedirs(outpath, exist_ok=True)
        fields = getattr(cfg, "fields", list(FIELD_REGISTRY.keys()))
        active = {k: FIELD_REGISTRY[k] for k in fields}
        os.makedirs(outpath, exist_ok=True)

        for i in range(windows):
            window = "{:02d}".format(i)
            tem_qmmm_path = f"{qmmmpath}/{window}/"
            win_data = {k: [] for k in active}

            for j in range(nframes):
                frame = "{:04d}".format(j)
                # --- 文件检查（同原来） ---
                tem_input = f"{tem_qmmm_path}/{frame}/{prefix}{frame}.out"
                err = self.check_qchem_error(tem_input)
                if err == -1:
                    print("File not found:", tem_input)
                    continue
                elif err == 1:
                    print("Qchem Job Fail:", tem_input)
                    continue

                tem_qmout = qchem_file()
                tem_qmout.molecule.check = True
                tem_qmout.external_charges.check = True
                tem_qmout.read_file(f"{tem_qmmm_path}/{frame}/{prefix}{frame}.inp")

                # 构造 context 供 loader 使用
                ctx = {
                    "cache_path": f"{cache_path}/{window}/{frame}/",
                    "qmout": tem_qmout,
                    "esp_file": f"{tem_qmmm_path}/{frame}/{prefix}{frame}.out.esp",
                    "efld_file": f"{tem_qmmm_path}/{frame}/{prefix}{frame}.out.efld",
                    "n_mm": len(tem_qmout.external_charges.mm_charge),
                }

                # 逐字段加载
                frame_data = {}
                skip = False
                for k, spec in active.items():
                    try:
                        frame_data[k] = spec["loader"](ctx)
                    except Exception as e:
                        print(f"Error loading {k} for {tem_input}: {e}")
                        skip = True
                        break
                if skip:
                    continue

                # MM ESP 尺寸一致性检查（仅在选了相关字段时）
                if "mm_esp" in frame_data or "mm_esp_grad" in frame_data:
                    n_esp = frame_data.get("mm_esp", frame_data.get("mm_esp_grad"))
                    if len(win_data.get("mm_esp", win_data.get("mm_esp_grad", []))) == 0:
                        ref_n = n_esp.shape[0]
                    elif n_esp.shape[0] != ref_n:
                        print("Skip frame due to inconsistent MM ESP size:", tem_input)
                        continue

                for k in active:
                    win_data[k].append(frame_data[k])

            # 保存每个 window
            for k, spec in active.items():
                arr = np.asarray(win_data[k])
                np.save(f"{outpath}/{spec['save_name']}_w{window}.npy", arr)
            del win_data

        # 从磁盘拼接 full
        for k, spec in active.items():
            parts = []
            for i in range(windows):
                window = "{:02d}".format(i)
                fpath = f"{outpath}/{spec['save_name']}_w{window}.npy"
                if os.path.exists(fpath):
                    parts.append(np.load(fpath))
            if parts:
                merged = np.concatenate(parts, axis=0)
                if "merge_post" in spec:
                    merged = spec["merge_post"](merged)
                np.save(f"{outpath}/full_{spec['save_name']}.npy", merged)
                del merged
            del parts



    def run_gas(self, cfg):
        qmmmpath = cfg.qmmmpath
        cache_path = cfg.cache_path
        outpath = cfg.outpath
        prefix = cfg.prefix
        windows = cfg.windows
        nframes = cfg.nframes
        os.makedirs(outpath, exist_ok=True)

        full_energy = []
        full_qm_grad = []
        full_qm_coords = []
        full_qm_type = None
        for i in range(windows):
            window = "{:02d}".format(i)
            tem_qmmm_path = f"{qmmmpath}/{window}/"

            win_energy = []
            win_qm_grad = []
            win_qm_coord = []

            for j in range(nframes):
                frame = "{:04d}".format(j)
                tem_cache_path = f"{cache_path}/{window}/{frame}/"
                tem_input = f"{tem_qmmm_path}/{frame}/{prefix}{frame}.out"
                tem_qmout = qchem_out_force()
                if self.check_qchem_error(tem_input) == -1:
                    print("File not found:", tem_input)
                    continue
                elif self.check_qchem_error(tem_input) == 1:
                    print("Qchem Job Fail:", tem_input)
                    continue
                tem_qmout.read_file(tem_input)
                if full_qm_type is None:
                    from chemlab.util.file_system import atom_charge_dict
                    atom_symbols = [atm[0] for atm in tem_qmout.molecule.carti]
                    qm_type = np.array([atom_charge_dict[sym] for sym in atom_symbols])
                    full_qm_type = qm_type
                tem_qm_coord = tem_qmout.molecule.xyz
                tem_energy = tem_qmout.ene
                tem_qm_grad = tem_qmout.force

                win_energy.append(tem_energy)
                win_qm_grad.append(tem_qm_grad)

                win_qm_coord.append(tem_qm_coord)

            from chemlab.util.unit import ENERGY,GRADIENT,DISTANCE


            win_qm_coord = np.asarray(win_qm_coord)
            win_energy = np.asarray(win_energy)
            win_qm_grad = np.asarray(win_qm_grad)
            win_energy = ENERGY(win_energy, "hartree").convert_to("kcal")
            win_qm_grad = GRADIENT(win_qm_grad, energy_unit="hartree", distance_unit="bohr").convert_to(
                {"energy": ("kcal", 1), "distance": ("ang", -1)}
            )

            np.save(f"{outpath}/qm_coord_w{window}.npy", win_qm_coord)
            np.save(f"{outpath}/energy_w{window}.npy", win_energy)
            np.save(f"{outpath}/qm_grad_w{window}.npy", win_qm_grad)


            full_energy.append(win_energy)
            full_qm_grad.append(win_qm_grad)
            full_qm_coords.append(win_qm_coord)
        full_energy = np.concatenate(full_energy, axis=0)
        np.save(f"{outpath}/full_energy.npy", full_energy)
        full_qm_grad = np.concatenate(full_qm_grad, axis=0)
        np.save(f"{outpath}/full_qm_grad.npy", full_qm_grad)
        full_qm_coords = np.concatenate(full_qm_coords, axis=0)
        np.save(f"{outpath}/full_qm_coords.npy", full_qm_coords)
        np.save(f"{outpath}/full_qm_type.npy", full_qm_type)