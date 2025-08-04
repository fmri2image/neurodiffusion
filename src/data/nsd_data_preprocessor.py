# src/data/nsd_data_preprocessor.py
import os
import numpy as np
import pandas as pd
import argparse
import scipy.io
from nsd_access import NSDAccess
from typing import List


class NSDPreprocessor:
    def __init__(self, nsd_root: str, output_dir: str, atlas_name: str = "streams"):
        self.nsda = NSDAccess(nsd_root)
        self.output_dir = output_dir
        self.atlas_name = atlas_name
        self.nsd_root = nsd_root

        os.makedirs(output_dir, exist_ok=True)

        self.sharedix = self._load_shared_indices()

    def _load_shared_indices(self):
        mat = scipy.io.loadmat(os.path.join(self.nsd_root, "nsddata/experiments/nsd/nsd_expdesign.mat"))
        return mat["sharedix"].flatten() - 1  # Convert from 1-based to 0-based

    def run(self, subject: str, overwrite: bool = False):
        print(f"[INFO] Preprocessing subject: {subject}")
        savedir = os.path.join(self.output_dir, subject)
        os.makedirs(savedir, exist_ok=True)

        # Load behavior (stimulus IDs)
        behs = pd.concat([
            self.nsda.read_behavior(subject=subject, session_index=i)
            for i in range(1, 38)
        ])

        stims_all = behs["73KID"].values - 1  # 0-based
        stims_unique = np.unique(stims_all)

        # Save stimulus lists if not exist
        stim_path = os.path.join(savedir, f"{subject}_stims_each.npy")
        stim_ave_path = os.path.join(savedir, f"{subject}_stims_ave.npy")

        if overwrite or not os.path.exists(stim_path):
            np.save(stim_path, stims_all)
            np.save(stim_ave_path, stims_unique)

        # Load all session betas
        print("[INFO] Loading beta weights...")
        betas_all = np.concatenate([
            self.nsda.read_betas(
                subject=subject,
                session_index=i,
                trial_index=[],
                data_type="betas_fithrf_GLMdenoise_RR",
                data_format="func1pt8mm"
            )
            for i in range(1, 38)
        ])

        # Read ROI mask atlas
        atlas = self.nsda.read_atlas_results(subject=subject, atlas=self.atlas_name, data_format="func1pt8mm")
        print(atlas)
        print(atlas[1].items())

        for roi, val in atlas[1].items():
            if val == 0:
                print(f"[SKIP] ROI '{roi}' is empty.")
                continue

            print(f"[ROI] Processing: {roi}")
            roi_mask = atlas[0].transpose([2, 1, 0]) == val
            betas_roi = betas_all[:, roi_mask]

            # Averaged over repeated stimuli
            betas_roi_ave = np.stack([
                np.mean(betas_roi[stims_all == stim], axis=0)
                for stim in stims_unique
            ])

            # Split into train/test (ALL)
            betas_tr = np.stack([betas_roi[i] for i, stim in enumerate(stims_all) if stim not in self.sharedix])
            betas_te = np.stack([betas_roi[i] for i, stim in enumerate(stims_all) if stim in self.sharedix])

            # Split into train/test (AVERAGED)
            betas_ave_tr = np.stack([betas_roi_ave[i] for i, stim in enumerate(stims_unique) if stim not in self.sharedix])
            betas_ave_te = np.stack([betas_roi_ave[i] for i, stim in enumerate(stims_unique) if stim in self.sharedix])

            # Save
            np.save(os.path.join(savedir, f"{subject}_{roi}_betas_tr.npy"), betas_tr)
            np.save(os.path.join(savedir, f"{subject}_{roi}_betas_te.npy"), betas_te)
            np.save(os.path.join(savedir, f"{subject}_{roi}_betas_ave_tr.npy"), betas_ave_tr)
            np.save(os.path.join(savedir, f"{subject}_{roi}_betas_ave_te.npy"), betas_ave_te)
            print(f"[SAVE] Finished saving ROI '{roi}' data.")


def main():
    parser = argparse.ArgumentParser(description="NSD Beta Preprocessor")
    parser.add_argument("--subject", type=str, required=True, help="Subject name (e.g., subj01)")
    parser.add_argument("--nsd_root", type=str, default="/mnt/datasets/nsd", help="Root path to NSD dataset")
    parser.add_argument("--output_dir", type=str, default="/mnt/outputs/fmri_features", help="Directory to save extracted features")
    parser.add_argument("--atlas", type=str, default="streams", help="Atlas name (e.g., streams)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files")

    args = parser.parse_args()

    processor = NSDPreprocessor(nsd_root=args.nsd_root, output_dir=args.output_dir, atlas_name=args.atlas)
    processor.run(subject=args.subject, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
