import numpy as np
from chemlab.scripts.base import Script
from chemlab.config.config_loader import ConfigBase


class PhaseLockDipoleConfig(ConfigBase):
    """
    Config for phase_lock_dipole script.
    """
    section_name = "phase_lock_dipole"


class PhaseLockDipole(Script):
    """
    Phase-lock transition dipoles and ESP charges for bright/dark states.
    """
    name = "phase_lock_dipole"
    config = PhaseLockDipoleConfig

    def run(self, cfg):
        # ======== Load data ========
        print(f"[phase_lock_dipole] Loading data from: {cfg.input}")
        data = np.load(f"{cfg.path}/{cfg.input}", allow_pickle=True)

        mu = data["trans_moms"]           # (nframes, nexc, 3)
        qex = data["esp_trans_density"]   # (nframes, nexc, natoms)
        R = data["coords"]                # (nframes, natoms, 3)

        nframes = mu.shape[0]
        print(f"[phase_lock_dipole] Loaded {nframes} frames")

        # ======== Center coordinates ========
        # Centering avoids origin-dependent effects on dipole calculation
        R0 = R - R.mean(axis=1, keepdims=True)

        # ======== Identify bright/dark states ========
        mu1 = mu[:, 0, :]
        mu2 = mu[:, 1, :]
        m1 = np.linalg.norm(mu1, axis=1)
        m2 = np.linalg.norm(mu2, axis=1)
        bright_is_S2 = (m2 > m1)

        n_bright_s2 = np.sum(bright_is_S2)
        print(f"[phase_lock_dipole] Bright state: S2 in {n_bright_s2}/{nframes} frames "
              f"({100*n_bright_s2/nframes:.1f}%)")

        # ======== Separate bright/dark ========
        bright_mu = np.where(bright_is_S2[:, None], mu2, mu1)
        dark_mu = np.where(bright_is_S2[:, None], mu1, mu2)
        bright_q = np.where(bright_is_S2[:, None], qex[:, 1, :], qex[:, 0, :])
        dark_q = np.where(bright_is_S2[:, None], qex[:, 0, :], qex[:, 1, :])

        # ======== Phase lock ========
        print("[phase_lock_dipole] Phase-locking bright state...")
        bright_mu_al, bright_q_al = self._phase_lock(bright_mu, bright_q, R0)

        print("[phase_lock_dipole] Phase-locking dark state...")
        dark_mu_al, dark_q_al = self._phase_lock(dark_mu, dark_q, R0)

        # ======== Save outputs ========
        out_path = cfg.out.rstrip("/") + "/"
        prefix = cfg.prefix

        np.save(f"{out_path}{prefix}bright_dipole.npy", bright_mu_al)
        np.save(f"{out_path}{prefix}dark_dipole.npy", dark_mu_al)
        np.save(f"{out_path}{prefix}bright_charge.npy", bright_q_al)
        np.save(f"{out_path}{prefix}dark_charge.npy", dark_q_al)
        np.save(f"{out_path}{prefix}bright_is_S2.npy", bright_is_S2.astype(np.int8))

        print(f"[phase_lock_dipole] Saved outputs to {out_path}")
        print(f"  - {prefix}bright_dipole.npy  shape={bright_mu_al.shape}")
        print(f"  - {prefix}dark_dipole.npy    shape={dark_mu_al.shape}")
        print(f"  - {prefix}bright_charge.npy  shape={bright_q_al.shape}")
        print(f"  - {prefix}dark_charge.npy    shape={dark_q_al.shape}")
        print(f"  - {prefix}bright_is_S2.npy   shape={bright_is_S2.shape}")
        print("[phase_lock_dipole] Done.")

    # -------------------------------------------------------
    # Helper: phase locking
    # -------------------------------------------------------
    @staticmethod
    def _phase_lock(mu_vec, q_atom, R_centered):
        """
        Make (mu_vec, q_atom) phase-consistent frame-by-frame.

        Uses the transition charge distribution to predict a dipole direction,
        then flips both μ and q if the predicted direction is opposite to
        a reference direction (from the first non-zero frame).

        Args:
            mu_vec: (nframes, 3) transition dipole moments
            q_atom: (nframes, natoms) ESP transition charges
            R_centered: (nframes, natoms, 3) centered coordinates

        Returns:
            (aligned_mu, aligned_q) with consistent phase
        """
        # Predict dipole from charge distribution: μ_pred = Σ_a q_a * R_a
        mu_pred = np.einsum("faj,fa->fj", R_centered, q_atom)  # (frames, 3)

        # Find reference direction from first non-zero predicted dipole
        ref = None
        for i in range(len(mu_pred)):
            if np.linalg.norm(mu_pred[i]) > 1e-12:
                ref = mu_pred[i].copy()
                break

        if ref is None:
            # All zero? Return as-is
            return mu_vec.copy(), q_atom.copy()

        out_mu = mu_vec.copy()
        out_q = q_atom.copy()

        n_flipped = 0
        for i in range(len(mu_pred)):
            if np.linalg.norm(mu_pred[i]) < 1e-12:
                continue
            if np.dot(ref, mu_pred[i]) < 0:
                out_mu[i] *= -1.0
                out_q[i] *= -1.0
                n_flipped += 1

        print(f"    Flipped {n_flipped}/{len(mu_pred)} frames for phase consistency")
        return out_mu, out_q