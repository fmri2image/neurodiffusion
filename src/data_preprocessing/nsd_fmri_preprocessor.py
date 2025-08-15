import os
import numpy as np
import pandas as pd
import argparse
import scipy.io
from nsd_access import NSDAccess


class NSDPreprocessor:
    def __init__(self, output_dir: str, nsd_root: str, atlas_name: str = "streams"):
        self.nsd_root = nsd_root
        self.nsda = NSDAccess(self.nsd_root)
        self.output_dir = output_dir
        self.atlas_name = atlas_name

        os.makedirs(output_dir, exist_ok=True)

    def run(self, subject: str):
        print(f"[INFO] Preprocessing subject: {subject}")
        savedir = os.path.join(self.output_dir, subject)
        os.makedirs(savedir, exist_ok=True)

        nsd_expdesign = scipy.io.loadmat(os.path.join(self.nsd_root, "nsddata/experiments/nsd/nsd_expdesign.mat"))
        sharedix = nsd_expdesign['sharedix'] - 1

        behs = pd.DataFrame()
        for i in range(1, 38):
            beh = self.nsda.read_behavior(subject=subject,
                                          session_index=i)
            behs = pd.concat((behs, beh))

        stims_unique = behs['73KID'].unique() - 1
        stims_all = behs['73KID'] - 1

        # Save stimulus lists if not exist
        stim_path = os.path.join(savedir, f"{subject}_stims_each.npy")
        stim_ave_path = os.path.join(savedir, f"{subject}_stims_ave.npy")

        if not os.path.exists(stim_path):
            np.save(stim_path, stims_all)
            np.save(stim_ave_path, stims_unique)

        # Load all session betas
        print("[INFO] Loading beta weights...")

        all_sessions = []

        for i in range(1, 38):
            print(i)
            beta_session = self.nsda.read_betas(subject=subject,
                                                session_index=i,
                                                trial_index=[],  # empty list as index means get all for this session
                                                data_type='betas_fithrf_GLMdenoise_RR',
                                                data_format='func1pt8mm')

            all_sessions.append(beta_session)

        betas_all = np.concatenate(all_sessions, axis=3)
        print(betas_all.shape)

        # Read ROI mask atlas
        atlas = self.nsda.read_atlas_results(subject=subject, atlas=self.atlas_name, data_format="func1pt8mm")
        print("atlas shape", len(atlas))
        print(len(atlas[0]))

        atlas_vol, atlas_map = atlas[0], atlas[1]
        if atlas_vol.shape != betas_all.shape[:3]:
            print("Inconsistency in Shape found between atlas_vol and betas_all first 3 dimensions")
            atlas_vol = atlas_vol.transpose(2, 1, 0)  # common NSD fix

        assert atlas_vol.shape == betas_all.shape[:3], f"{atlas_vol.shape} vs {betas_all.shape[:3]}"

        for roi, val in atlas_map.items():
            if val == 0:
                print('SKIP')
                continue
            else:
                mask = (atlas_vol == val)
                betas_roi = betas_all[mask, :].T

            # Averaging for each stimulus
            betas_roi_ave = []
            for stim in stims_unique:
                stim_mean = np.mean(betas_roi[stims_all == stim, :], axis=0)
                betas_roi_ave.append(stim_mean)
            betas_roi_ave = np.stack(betas_roi_ave)
            print(betas_roi_ave.shape)

            # Train/Test Split
            # ALLDATA
            betas_tr = []
            betas_te = []

            for idx, stim in enumerate(stims_all):
                if stim in sharedix:
                    betas_te.append(betas_roi[idx, :])
                else:
                    betas_tr.append(betas_roi[idx, :])

            betas_tr = np.stack(betas_tr)
            betas_te = np.stack(betas_te)

            # AVERAGED DATA
            betas_ave_tr = []
            betas_ave_te = []
            for idx, stim in enumerate(stims_unique):
                if stim in sharedix:
                    betas_ave_te.append(betas_roi_ave[idx, :])
                else:
                    betas_ave_tr.append(betas_roi_ave[idx, :])
            betas_ave_tr = np.stack(betas_ave_tr)
            betas_ave_te = np.stack(betas_ave_te)

            # Save
            np.save(os.path.join(savedir, f"{subject}_{roi}_betas_tr.npy"), betas_tr)
            np.save(os.path.join(savedir, f"{subject}_{roi}_betas_te.npy"), betas_te)
            np.save(os.path.join(savedir, f"{subject}_{roi}_betas_ave_tr.npy"), betas_ave_tr)
            np.save(os.path.join(savedir, f"{subject}_{roi}_betas_ave_te.npy"), betas_ave_te)
            print(f"[SAVE] Finished saving ROI '{roi}' data.")


def main():
    parser = argparse.ArgumentParser(description="NSD Betas Preprocessor")
    parser.add_argument("--subject", type=str, required=True, help="Subject name (e.g., subj01)")
    parser.add_argument("--output_dir", type=str, default="/mnt/outputs/fmri_features",
                        help="Directory to save extracted features")
    parser.add_argument("--atlas", type=str, default="streams", help="Atlas name (e.g., streams)")
    parser.add_argument("--nsd_root", type=str)

    args = parser.parse_args()

    processor = NSDPreprocessor(output_dir=args.output_dir, atlas_name=args.atlas, nsd_root=args.nsd_root)
    processor.run(subject=args.subject)


if __name__ == "__main__":
    main()