import os
import argparse
from typing import List, Optional, Tuple, Sequence

import numpy as np
from tqdm import tqdm


ALLOWED_FEATURES = ["inception", "alexnet5", "alexnet12", "alexnet18", "clip", "clip_h6", "clip_h12"]


class IdentificationAccuracyEvaluator:
    """
    Compute identification accuracy from extracted features of:
      - original images:  {imgid:05}_org_{feature}.npy
      - recon samples:    {imgid:05}_{rep:03}_{feature}.npy

    Expects features under:
      {output_dir}/eval_extracted_features/{method}/{subject}/
    """

    def __init__(
        self,
        subject: str,
        method: str,
        feature: str,                    # one of: ALLOWED_FEATURES
        output_dir: str = "/mnt/outputs",
        n_images: int = 982,
        n_rep: Optional[int] = None,     # if None -> auto-detect
        img_range: Optional[Tuple[int, int]] = None,  # e.g. (0, 981)
        strict: bool = True,
        chunk_size: int = 64,            # batch size when comparing against all "fake" images
    ):
        self.subject = subject
        self.method = method
        self.feature = feature
        self.output_dir = output_dir
        self.base_dir = os.path.join(output_dir, "eval_extracted_features", method, subject)

        if img_range is None:
            self.indices = list(range(n_images))
        else:
            s, e = img_range
            if s > e:
                s, e = e, s
            self.indices = list(range(s, e + 1))

        # position lookup: global index -> position in our arrays
        self.pos_by_idx = {idx: pos for pos, idx in enumerate(self.indices)}

        self.n_rep = n_rep
        self.strict = strict
        self.chunk_size = max(int(chunk_size), 1)

        os.makedirs(self.base_dir, exist_ok=True)

        if self.n_rep is None:
            self.n_rep = self._autodetect_reps()
            if self.n_rep is None:
                raise FileNotFoundError(
                    f"Could not auto-detect number of reps for feature '{self.feature}'. "
                    f"Ensure files like 00000_000_{self.feature}.npy exist."
                )

    # ---------------- Paths ----------------

    def _org_path(self, imgid: int) -> str:
        return os.path.join(self.base_dir, f"{imgid:05}_org_{self.feature}.npy")

    def _rep_path(self, imgid: int, rep: int) -> str:
        return os.path.join(self.base_dir, f"{imgid:05}_{rep:03}_{self.feature}.npy")

    # --------------- Autodetect reps ---------------

    def _autodetect_reps(self) -> Optional[int]:
        # find first image that exists, then count consecutive rep files
        for idx in self.indices:
            org = self._org_path(idx)
            if os.path.exists(org):
                rep = 0
                while os.path.exists(self._rep_path(idx, rep)):
                    rep += 1
                return rep if rep > 0 else None
        return None

    # --------------- Loading ----------------

    def _load_feat_org(self, imgid: int) -> Optional[np.ndarray]:
        p = self._org_path(imgid)
        if not os.path.exists(p):
            if self.strict:
                raise FileNotFoundError(f"Missing original feature: {p}")
            return None
        return np.load(p, allow_pickle=False).astype(np.float32).ravel()

    def _load_feat_gen(self, imgid: int) -> List[np.ndarray]:
        feats = []
        for rep in range(self.n_rep):
            p = self._rep_path(imgid, rep)
            if not os.path.exists(p):
                if self.strict:
                    raise FileNotFoundError(f"Missing generated feature: {p}")
                else:
                    continue
            feats.append(np.load(p, allow_pickle=False).astype(np.float32).ravel())
        return feats

    # --------------- Correlation helpers (vectorized) ----------------

    @staticmethod
    def _center_and_norm(x: np.ndarray) -> tuple[np.ndarray, float]:
        """Return centered vector and its L2 norm."""
        xc = x - x.mean()
        n = np.linalg.norm(xc)
        return xc, float(n)

    @staticmethod
    def _corrs_v_against_reps(v: np.ndarray, reps: np.ndarray) -> np.ndarray:
        """
        Compute correlations between vector v (D,) and reps (R, D) in a vectorized way.
        Returns (R,) of Pearson correlations.
        """
        v_c, v_n = IdentificationAccuracyEvaluator._center_and_norm(v)
        if v_n == 0.0:
            return np.zeros(reps.shape[0], dtype=np.float32)
        reps_c = reps - reps.mean(axis=1, keepdims=True)             # (R, D)
        reps_n = np.linalg.norm(reps_c, axis=1)                      # (R,)
        denom = v_n * reps_n
        dots = reps_c @ v_c                                          # (R,)
        out = np.zeros_like(dots, dtype=np.float32)
        mask = denom > 0
        out[mask] = dots[mask] / denom[mask]
        return out.astype(np.float32)

    @staticmethod
    def _mean_corr_v_vs_list_reps(v: np.ndarray, reps_list: List[np.ndarray]) -> float:
        """
        Mean correlation between v and each row in every matrix in reps_list (each with shape (R, D)).
        Here, reps_list is usually len==1 (the "true" image) or many (fakes batched).
        For convenience we support both single and batched: stack if needed.
        """
        if len(reps_list) == 0:
            return np.nan
        reps = np.vstack(reps_list)          # (B*R, D) or (R, D) if B==1
        corrs = IdentificationAccuracyEvaluator._corrs_v_against_reps(v, reps)
        return float(np.nanmean(corrs))

    # --------------- Accuracy ----------------

    def compute_accuracy(self) -> float:
        # Load everything into memory for this feature (keeps IO minimal for comparisons)
        print(f"[{self.feature}] Loading features...")
        feat_orgs: List[Optional[np.ndarray]] = []
        feat_gens: List[List[np.ndarray]] = []

        for imgid in tqdm(self.indices, desc=f"load-{self.feature}", unit="img", leave=False):
            org = self._load_feat_org(imgid)
            gens = self._load_feat_gen(imgid)

            if self.strict:
                if org is None or len(gens) == 0:
                    raise RuntimeError(f"Missing features for imgid={imgid}")
            else:
                if org is None or len(gens) == 0:
                    feat_orgs.append(None)
                    feat_gens.append([])
                    continue

            feat_orgs.append(org)
            feat_gens.append(gens)

        valid_indices = [i for i, o in zip(self.indices, feat_orgs) if o is not None]

        print(f"[{self.feature}] Computing correlations & accuracy...")
        acc_all = []

        for i in tqdm(valid_indices, desc=f"identify-{self.feature}", unit="img", leave=False):
            pi = self.pos_by_idx[i]
            v_org = feat_orgs[pi]
            # mean corr vs own reps
            r_true_mean = self._mean_corr_v_vs_list_reps(v_org, feat_gens[pi])

            correct, total = 0, 0

            # Compare against all other images in chunks to save memory
            others = [j for j in valid_indices if j != i]
            for start in range(0, len(others), self.chunk_size):
                batch_idxs = others[start:start + self.chunk_size]
                batch_pos = [self.pos_by_idx[j] for j in batch_idxs]

                # stack reps for each j into a single list (we only need mean over reps per j)
                # We'll compute means per j by splitting the flat correlations back, but
                # a simple loop over this small chunk is already cheap.
                for pj in batch_pos:
                    r_fake_mean = self._mean_corr_v_vs_list_reps(v_org, feat_gens[pj])
                    if not np.isnan(r_fake_mean):
                        correct += (r_true_mean > r_fake_mean)
                        total += 1

            if total > 0:
                acc_all.append(correct / total)

        if not acc_all:
            raise RuntimeError(f"[{self.feature}] No valid images to evaluate.")
        return float(np.mean(acc_all))

    def run(self) -> float:
        acc = self.compute_accuracy()
        print(f"{self.subject}_{self.feature}:\t ACC = {acc:.3f}")
        return acc


# --------------- CLI helpers ----------------

def _parse_range(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    parts = s.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("img-range must be two ints: START END")
    a, b = int(parts[0]), int(parts[1])
    return (min(a, b), max(a, b))


def _parse_features(fs: Sequence[str]) -> List[str]:
    if len(fs) == 1 and fs[0].lower() == "all":
        return ALLOWED_FEATURES
    out = [f.lower() for f in fs]
    bad = [f for f in out if f not in ALLOWED_FEATURES]
    if bad:
        raise ValueError(f"Unknown features {bad}; allowed: {ALLOWED_FEATURES}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Compute identification accuracy from extracted features.")
    parser.add_argument("--subject", required=True, type=str, help="subj01 | subj02 | subj05 | subj07 ...")
    parser.add_argument("--method", required=True, type=str, help="e.g., kernelridgeregression or your decoder tag")
    parser.add_argument("--features", nargs="+", default=["all"],
                        help="One or more of: "
                             "'inception' 'alexnet5' 'alexnet12' 'alexnet18' 'clip' 'clip_h6' 'clip_h12' "
                             "or 'all' to run all.")
    parser.add_argument("--output-dir", type=str, default="/mnt/outputs")
    parser.add_argument("--n-images", type=int, default=982)
    parser.add_argument("--n-rep", type=int, default=None)
    parser.add_argument("--img-range", type=str, default=None, help="Optional range 'START END' or 'START,END'.")
    parser.add_argument("--non-strict", action="store_true",
                        help="Skip images/reps with missing files instead of raising.")
    parser.add_argument("--chunk-size", type=int, default=64, help="Batch size for fake comparisons.")
    args = parser.parse_args()

    feats = _parse_features(args.features)
    rng = _parse_range(args.img_range)

    results = {}
    for feat in feats:
        evaluator = IdentificationAccuracyEvaluator(
            subject=args.subject,
            method=args.method,
            feature=feat,
            output_dir=args.output_dir,
            n_images=args.n_images,
            n_rep=args.n_rep,
            img_range=rng,
            strict=not args.non_strict,
            chunk_size=args.chunk_size,
        )
        results[feat] = evaluator.run()

    print("\n=== Identification accuracy summary ===")
    for f in ALLOWED_FEATURES:
        if f in results:
            print(f"{f:10s}: {results[f]:.3f}")


if __name__ == "__main__":
    main()
