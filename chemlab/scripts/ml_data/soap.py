from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase
from chemlab.util.file_system import NUM2ELEMENT
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
class BuildSOAPConfig(ConfigBase):
    section_name = "build_soap"

class BuildSOAP(Script):
    name = "build_soap"
    config = BuildSOAPConfig
    def run(self,cfg):
        windows = cfg.windows
        npy_path = cfg.npy_path
        qm_type_file = cfg.qm_type
        qm_type = np.load(os.path.join(npy_path,qm_type_file))
        elements = [NUM2ELEMENT[int(i)] for i in qm_type]
        out_path=cfg.out_path
        r_cut = cfg.r_cut
        n_max = cfg.n_max
        l_max = cfg.l_max
        sigma = cfg.sigma
        n_select = cfg.n_select
        random_seed = cfg.random_seed
        test_set = cfg.test_set
        test_set = np.load(os.path.join(npy_path,test_set))
        global_to_window,window_sizes = build_global_to_window_mapping(npy_path,windows)
        soap = SOAP(
            species=elements,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            periodic=False,
            average="outer"
        )
        all_selected_global = []
        for i in range(windows):
            coords = np.load(os.path.join(npy_path,f"qm_coord_w{i:02d}.npy"))
            soap_features = []
            for j, coord in enumerate(coords):
                soap_features.append(self.single_frame_soap(coord,qm_type,soap))

            soap_features = np.array(soap_features)  # (n_structures, n_features)


            #np.save(f"{out_path}/soap_features_w{i:02d}.npy", soap_features)
            n_structures = soap_features.shape[0]
            n_features = soap_features.shape[1]
            local_exclude = get_local_exclude_for_window(i, test_set, global_to_window)
            all_indices = np.arange(n_structures)
            available_indices = np.setdiff1d(all_indices, local_exclude)
            available_soap = soap_features[available_indices]


            kmeans = KMeans(n_clusters=n_select, random_state=random_seed, n_init=10)
            labels = kmeans.fit_predict(available_soap)
            selected_in_available = []
            for cluster_id in range(n_select):
                cluster_mask = (labels == cluster_id)
                cluster_indices = np.where(cluster_mask)[0]

                if len(cluster_indices) == 0:
                    continue

                cluster_structures = available_soap[cluster_mask]
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_structures - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_in_available.append(closest_idx)

            selected_in_available = np.array(selected_in_available)
            selected_local = available_indices[selected_in_available]
            window_start_global = sum(window_sizes[:i])
            selected_global = selected_local + window_start_global
            all_selected_global.extend(selected_global)

        all_selected_global = np.array(all_selected_global)
        np.savez(f"{out_path}/soap_{windows*n_select}_split.npz", idx_train=all_selected_global, idx_val=test_set["idx_val"],idx_test=test_set["idx_testt"])
    @staticmethod
    def single_frame_soap(coord,qm_type,soap):
        atoms = Atoms(numbers=qm_type, positions=coord)
        soap_vec = soap.create(atoms)
        return soap_vec


def build_global_to_window_mapping(npy_path, windows):
    global_to_window = {}
    window_sizes = []
    global_idx = 0
    for window_id in range(windows):
        coords = np.load(os.path.join(npy_path, f"qm_coord_w{window_id:02d}.npy"))
        n_structures = len(coords)
        window_sizes.append(n_structures)

        for local_idx in range(n_structures):
            global_to_window[global_idx] = (window_id, local_idx)
            global_idx += 1

    return global_to_window, window_sizes


def get_local_exclude_for_window(window_id, test_set, global_to_window):
    exclude_global = np.concatenate([test_set['idx_test'], test_set['idx_val']])
    local_exclude = []
    for global_idx in exclude_global:
        win_id, local_idx = global_to_window[global_idx]
        if win_id == window_id:
            local_exclude.append(local_idx)
    return np.array(local_exclude)