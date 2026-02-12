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
    '''
    基于给定的描述符，聚类选出结构（要输入给定的test/val)
    '''
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
        method = cfg.method
        global_to_window,window_sizes = build_global_to_window_mapping(npy_path,windows)

        if method == "soap":
            descriptor = SOAP(
                species=elements,
                r_cut=r_cut,
                n_max=n_max,
                l_max=l_max,
                sigma=sigma,
                periodic=False,
                average="outer"
            )
        
        all_selected_global = []
        results=[]
        for i in range(windows):
            coords = np.load(os.path.join(npy_path,f"qm_coord_w{i:02d}.npy"))
            features = []
            for j, coord in enumerate(coords):
                features.append(self.single_frame_soap(coord,qm_type,descriptor))

            features = np.array(features)  # (n_structures, n_features)


            #np.save(f"{out_path}/soap_features_w{i:02d}.npy", features)
            n_structures = features.shape[0]
            n_features = features.shape[1]
            local_exclude = get_local_exclude_for_window(i, test_set, global_to_window)
            all_indices = np.arange(n_structures)
            available_indices = np.setdiff1d(all_indices, local_exclude)
            available_soap = features[available_indices]


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
            rng = np.random.default_rng(random_seed)
            random_in_available = rng.choice(len(available_indices),
                                             size=len(selected_in_available),
                                             replace=False)
            random_local = available_indices[random_in_available]


            cs, ds = compute_metrics(features, selected_local)
            cr, dr = compute_metrics(features, random_local)
            results.append({'cs': cs, 'cr': cr, 'ds': ds, 'dr': dr})

            print(f"W{i}: SOAP cov={cs:.3f} div={ds:.3f}, Random cov={cr:.3f} div={dr:.3f}")

            # 可视化前3个window
            if i < 3:
                plot_comparison(features, selected_local, random_local, i, out_path)

        all_selected_global = np.array(all_selected_global)
        np.savez(f"{out_path}/soap_{windows*n_select}_split.npz", idx_train=all_selected_global, idx_val=test_set["idx_val"],idx_test=test_set["idx_test"])
        plot_all_summary(results, out_path)
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


def compute_metrics(soap_features, selected_indices):
    from sklearn.metrics import pairwise_distances
    selected = soap_features[selected_indices]
    coverage = pairwise_distances(soap_features, selected).min(axis=1).mean()
    if len(selected) > 1:
        dists = pairwise_distances(selected)
        mask = np.triu(np.ones(dists.shape), k=1).astype(bool)
        diversity = dists[mask].mean()
    else:
        diversity = 0.0
    return coverage, diversity


def plot_comparison(soap_features, soap_idx, rand_idx, window_id, out_path):
    """绘制单个window的对比图"""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    soap_2d = pca.fit_transform(soap_features)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # SOAP
    ax1.scatter(soap_2d[:, 0], soap_2d[:, 1], c='gray', s=10, alpha=0.3)
    ax1.scatter(soap_2d[soap_idx, 0], soap_2d[soap_idx, 1], c='red', s=60, edgecolors='black')
    ax1.set_title(f'SOAP - Window {window_id}')

    # Random
    ax2.scatter(soap_2d[:, 0], soap_2d[:, 1], c='gray', s=10, alpha=0.3)
    ax2.scatter(soap_2d[rand_idx, 0], soap_2d[rand_idx, 1], c='blue', s=60, edgecolors='black')
    ax2.set_title(f'Random - Window {window_id}')

    plt.tight_layout()
    plt.savefig(f"{out_path}/w{window_id:02d}_comparison.png", dpi=120)
    plt.close()


def plot_all_summary(results, out_path):
    """绘制所有windows的总结"""
    import matplotlib.pyplot as plt

    cov_s = [r['cs'] for r in results]
    cov_r = [r['cr'] for r in results]
    div_s = [r['ds'] for r in results]
    div_r = [r['dr'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = range(len(results))
    w = 0.35

    ax1.bar([i - w / 2 for i in x], cov_s, w, label='SOAP', alpha=0.8)
    ax1.bar([i + w / 2 for i in x], cov_r, w, label='Random', alpha=0.8)
    ax1.set_ylabel('Coverage (↓)')
    ax1.set_title('Coverage Comparison')
    ax1.legend()

    ax2.bar([i - w / 2 for i in x], div_s, w, label='SOAP', alpha=0.8)
    ax2.bar([i + w / 2 for i in x], div_r, w, label='Random', alpha=0.8)
    ax2.set_ylabel('Diversity (↑)')
    ax2.set_title('Diversity Comparison')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{out_path}/summary.png", dpi=120)
    plt.close()