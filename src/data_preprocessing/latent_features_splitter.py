import numpy as np
import scipy.io
from tqdm import tqdm
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LatentsSplitter")


class LatentsSplitter:
    """Class to split stimuli features into train and test sets based on NSD sharedix."""

    def __init__(self, featname, use_stim, subject, nsd_dir, output_dir):
        """
        Initialize the stimuli splitter.

        Args:
            featname (str): Feature type ('init_latent' or 'c').
            use_stim (str): Stimulus type ('ave' or 'each').
            subject (str): Subject ID (e.g., 'subj01').
            nsd_dir (str): Directory containing NSD data.
            output_dir (str): Directory to save train/test splits.
        """
        self.featname = featname
        self.use_stim = use_stim
        self.subject = subject
        self.nsd_dir = nsd_dir
        self.output_dir = output_dir

        # Load NSD experiment design
        self.nsd_expdesign = scipy.io.loadmat(
            os.path.join(nsd_dir, 'nsddata/experiments/nsd/nsd_expdesign.mat'))

        self.sharedix = self.nsd_expdesign['sharedix'] - 1  # Convert to 0-based indexing

    def load_stimuli(self):
        """Load stimuli indices."""
        try:
            stims_path = os.path.join(self.output_dir, f'fmri_features/{self.subject}',
                                      f'{self.subject}_stims_{self.use_stim}.npy')
            stims = np.load(stims_path)
            return stims
        except Exception as e:
            logger.error(f"Error loading stimuli from {stims_path}: {str(e)}")
            raise

    def split_latent_features(self):
        """Split features into train and test sets."""
        logger.info(f"Splitting features for {self.subject}, featname: {self.featname}, use_stim: {self.use_stim}")
        stims = self.load_stimuli()
        feats = []
        tr_idx = np.zeros(len(stims))

        for idx, s in tqdm(enumerate(stims), total=len(stims), desc="Processing stimuli"):
            if s in self.sharedix:
                tr_idx[idx] = 0  # Test set (sharedix)
            else:
                tr_idx[idx] = 1  # Train set
            feat_path = os.path.join(self.output_dir, f"latent_features/extracted_latents/{self.featname}/{self.subject}/{s:06}.npy")
            if not os.path.exists(feat_path):
                logger.warning(f"Feature file {feat_path} not found, skipping.")
                continue
            feat = np.load(feat_path)
            feats.append(feat)

        if not feats:
            raise ValueError("No features loaded. Check feature files in {self.feat_dir}.")

        feats = np.stack(feats)

        splits_path = os.path.join(self.output_dir, f"latent_features/splitted_latents/{self.subject}/")
        os.makedirs(splits_path, exist_ok=True)

        feats_tr = feats[tr_idx == 1, :]
        feats_te = feats[tr_idx == 0, :]

        tr_idx_path = os.path.join(splits_path, f"{self.subject}_stims_tridx.npy")

        feats_tr_path = os.path.join(splits_path, f"{self.subject}_{self.use_stim}_{self.featname}_tr.npy")
        feats_te_path = os.path.join(splits_path, f"{self.subject}_{self.use_stim}_{self.featname}_te.npy")

        np.save(tr_idx_path, tr_idx)
        np.save(feats_tr_path, feats_tr)
        np.save(feats_te_path, feats_te)

        logger.info(f"Saved train/test indices to {tr_idx_path}")
        logger.info(f"Saved train features to {feats_tr_path}")
        logger.info(f"Saved test features to {feats_te_path}")

        return tr_idx_path, feats_tr_path, feats_te_path


def main():
    parser = argparse.ArgumentParser(description="Split stimuli features into train and test sets.")
    parser.add_argument("--featname", type=str, required=True, help="Feature type (init_latent or c_topk_captions)")
    parser.add_argument("--use_stim", type=str, required=True, choices=['ave', 'each'],
                        help="Stimulus type (ave or each)")
    parser.add_argument("--subject", type=str, required=True, choices=['subj01', 'subj02', 'subj03', 'subj04', 'subj05',
                        'subj06', 'subj07', 'subj08'], help="Subject ID (e.g., subj01)")
    parser.add_argument("--nsd_dir", type=str, default="/mnt/dataset/nsd", help="NSD data directory")
    parser.add_argument("--output_dir", type=str, default="/mnt/outputs", help="Output directory")

    args = parser.parse_args()

    splitter = LatentsSplitter(
        featname=args.featname,
        use_stim=args.use_stim,
        subject=args.subject,
        nsd_dir=args.nsd_dir,
        output_dir=args.output_dir
    )
    splitter.split_latent_features()


if __name__ == "__main__":
    main()
