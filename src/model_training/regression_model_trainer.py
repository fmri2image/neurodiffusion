import os
import argparse
import logging
from time import perf_counter
from functools import partial
from abc import ABC, abstractmethod

import numpy as np
import joblib
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge

from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.metrics.pairwise import pairwise_distances

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RegressionModel")


# ---------- Base Class ----------
class RegressionModel(ABC):
    """Base class for regression models mapping fMRI to stimuli features."""

    def __init__(self, target, roi, subject, fmri_dir, feat_dir, models_dir, betas_flag, use_pca):
        """
        Args:
            target (str): Feature type (e.g., 'init_latent', 'c_topk_captions').
            roi (list|str): Region(s) of interest.
            subject (str): Subject ID (e.g., 'subj01').
            fmri_dir (str): Directory containing fMRI data.
            feat_dir (str): Directory containing train/test latent feature splits.
            models_dir (str): Directory to save trained models and predictions.
            betas_flag (str): 'each' or 'ave' for training betas.
            use_pca (bool): Whether to use PCA in the pipeline.
        """
        self.target = target
        # normalize roi to a list
        self.roi = roi if isinstance(roi, list) else [roi]
        self.subject = subject
        self.fmri_dir = os.path.join(fmri_dir, subject)
        self.feat_dir = os.path.join(feat_dir, f"splitted_latents/{subject}")
        self.models_dir = os.path.join(models_dir, subject)
        self.model = None
        self.backend = set_backend("torch_cuda" if 'mlp' in self.__class__.__name__.lower() else "numpy",
                                   on_error="warn")
        self.betas_flag = betas_flag
        self.use_pca = use_pca

    def load_data(self):
        """Load fMRI and feature data for training and testing."""
        try:
            t0 = perf_counter()
            X_list, Xte_list = [], []

            # Progress over ROIs
            for croi in tqdm(self.roi, desc=f"[{self.subject}] Loading ROIs", unit="roi", leave=False):
                if self.betas_flag == 'each':
                    X_list.append(
                        np.load(os.path.join(self.fmri_dir, f"{self.subject}_{croi}_betas_tr.npy")).astype("float32"))
                else:
                    X_list.append(
                        np.load(os.path.join(self.fmri_dir, f"{self.subject}_{croi}_betas_ave_tr.npy")).astype(
                            "float32"))

                Xte_list.append(
                    np.load(os.path.join(self.fmri_dir, f"{self.subject}_{croi}_betas_ave_te.npy")).astype("float32"))

            X = np.hstack(X_list)
            X_te = np.hstack(Xte_list)

            # Features
            tr_tag = 'each' if self.betas_flag == 'each' else 'ave'
            if tr_tag == 'ave':
                feat_tr_path = os.path.join(self.feat_dir, f"{self.subject}_{tr_tag}_{self.target}_tr.npy")
            else:
                feat_tr_path = os.path.join(self.feat_dir, f"{self.subject}_{tr_tag}_{self.target}_tr.npy")

            feat_te_path = os.path.join(self.feat_dir, f"{self.subject}_ave_{self.target}_te.npy")

            Y = np.load(feat_tr_path).astype("float32").reshape(X.shape[0], -1)
            Y_te = np.load(feat_te_path).astype("float32").reshape(X_te.shape[0], -1)

            logger.info(f"[{self.subject}] Loaded data in {perf_counter() - t0:.2f}s "
                        f"(X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape})")

            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"Mismatch between fMRI ({X.shape[0]}) and features ({Y.shape[0]}) samples.")

            return X, Y, X_te, Y_te
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    @abstractmethod
    def fit(self, X, y):
        """Fit the regression model. Must be implemented by subclasses."""
        pass

    def save_model(self):
        """Save the trained model (full pipeline) to models_dir."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        # If a subclass already saved and recorded the path, just return it
        if getattr(self, "_saved_model_path", None):
            return self._saved_model_path

        roi_str = "_".join(self.roi)
        pca_tag = f"pca{getattr(self, 'pca_dim', '')}" if getattr(self, "use_pca", False) else "nopca"

        model_name = (
            f"{self.__class__.__name__.lower()}_{pca_tag}_{self.target}_{roi_str}_"
            f"fmri_{self.betas_flag}_{self.subject}.joblib"
        )
        os.makedirs(self.models_dir, exist_ok=True)
        output_path = os.path.join(self.models_dir, model_name)
        joblib.dump(self.model, output_path, compress=3)
        logger.info(f"Saved model to {output_path}")

        self._saved_model_path = output_path
        return output_path

    def predict(self, X):
        """Default predict: pipeline-only. Subclasses may override (e.g., PCA inverse)."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)

    def predict_and_evaluate(self, X_te, Y_te):
        """Predict on test data and compute correlation score."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        # Use the overridable predict(), not self.model.predict()
        Y_hat = self.predict(X_te)

        # himalaya.scoring.correlation_score expects shape (n_targets, n_samples)
        rs = correlation_score(Y_te.T, Y_hat.T)
        mean_rs = np.mean(rs.cpu().numpy() if hasattr(rs, 'cpu') else rs)
        logger.info(f"Prediction correlation score: {mean_rs:.3f}")

        # Save predicted scores
        class_name = self.__class__.__name__.lower()
        roi_str = "_".join(self.roi)
        pca_tag = f"pca{getattr(self, 'pca_dim', '')}" if getattr(self, "use_pca", False) else "nopca"
        betas_tag = f"{self.betas_flag}"

        # base directory (keep kernel subfolder for KRR)
        base_dir_parts = [
            self.feat_dir,
            "prediction_scores",
            class_name,
        ]
        if class_name == "kernelridgeregression":
            base_dir_parts.append(f"{self.ridge_kernel}_kernel")
        base_dir = os.path.join(*base_dir_parts)

        # file name now includes betas + pca tags
        fname = f"{self.subject}_{roi_str}_scores_{self.target}_{betas_tag}_{pca_tag}.npy"
        scores_path = os.path.join(base_dir, fname)

        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        np.save(scores_path, Y_hat)
        logger.info(f"Saved predicted scores to {scores_path}")

        return Y_hat, mean_rs

    def train(self):
        """Train the model, save it, and evaluate on test data."""
        X, Y, X_te, Y_te = self.load_data()

        t_fit = perf_counter()
        self.fit(X, Y)
        logger.info(f"[{self.subject}] Fit {self.__class__.__name__} in {perf_counter() - t_fit:.2f}s")

        t_save = perf_counter()
        model_path = self.save_model()
        logger.info(f"[{self.subject}] Saved model in {perf_counter() - t_save:.2f}s → {model_path}")

        t_pred = perf_counter()
        scores, mean_rs = self.predict_and_evaluate(X_te, Y_te)
        logger.info(f"[{self.subject}] Pred+Eval in {perf_counter() - t_pred:.2f}s (mean corr={mean_rs:.3f})")

        return model_path, scores, mean_rs


# ---------- Models ----------
class KernelRidgeRegression(RegressionModel):
    """Kernel-based ridge regression model for non-linear fMRI-to-feature mapping."""

    def __init__(self, target, roi, subject, fmri_dir, feat_dir, models_dir,
                 betas_flag, use_pca, ridge_kernel, pca_dim=1024):
        super().__init__(target, roi, subject, fmri_dir, feat_dir, models_dir, betas_flag, use_pca)
        self.pca_dim = pca_dim
        self.ridge_kernel = ridge_kernel
        self.pca_model = None
        self.model = None
        self._saved_model_path = None

    @staticmethod
    def _median_gamma(X, max_samples=2000):
        n = X.shape[0]
        rng = np.random.RandomState(0)
        Xs = X if n <= max_samples else X[rng.choice(n, size=max_samples, replace=False)]
        d = pairwise_distances(Xs, metric="euclidean")
        md = np.median(d[np.triu_indices_from(d, k=1)])
        if not np.isfinite(md) or md <= 0:
            md = 1.0
        return 1.0 / (2.0 * (md ** 2))

    def fit(self, X, y):
        logger.info(
            f"Fitting KRR target={self.target}, ROIs={self.roi}, subject={self.subject}, "
            f"PCA={'on' if self.use_pca else 'off'}({self.pca_dim}), kernel={self.ridge_kernel}"
        )

        t0 = perf_counter()

        scaler = StandardScaler(copy=False)
        if self.ridge_kernel == "rbf":
            gamma = self._median_gamma(X)
            krr = KernelRidge(kernel="rbf", alpha=1e-3)
            logger.info(f"RBF gamma (median heuristic): {gamma:.3e}")
        else:
            krr = KernelRidge(kernel=self.ridge_kernel, alpha=1e-3)

        self.model = make_pipeline(scaler, krr)

        # Optionally compress Y (targets)
        if self.use_pca:
            n_samples, n_features = y.shape
            if self.pca_dim > min(n_samples, n_features):
                self.pca_dim = min(n_samples, n_features)
                logger.warning(f"Adjusted pca_dim to {self.pca_dim} for Y shape {y.shape}.")

            self.pca_model = PCA(
                n_components=self.pca_dim,
                svd_solver="randomized",
                random_state=42,
            )
            logger.info(f"Fitting PCA on Y {y.shape} → {self.pca_dim}")
            y_tr = self.pca_model.fit_transform(y)
            cumvar = float(self.pca_model.explained_variance_ratio_.cumsum()[-1])
            logger.info(f"PCA retained variance: {cumvar:.4f}")
        else:
            y_tr = y

        logger.info("Fitting KernelRidge on (X, transformed Y)")
        self.model.fit(X, y_tr)
        logger.info(f"KernelRidge fit done in {perf_counter() - t0:.2f}s")

        # Save artifacts
        roi_str = "_".join(self.roi)
        pca_tag = f"pca{self.pca_dim}" if self.use_pca else "nopca"
        base_name = f"kernelridgeregression_{self.ridge_kernel}_{pca_tag}_{self.betas_flag}_{self.target}_{roi_str}_{self.subject}"

        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, f"{base_name}.joblib")
        joblib.dump(self.model, model_path, compress=3)
        self._saved_model_path = model_path
        logger.info(f"Saved model pipeline to {model_path}")

        if self.use_pca:
            pca_path = os.path.join(self.models_dir, f"{base_name}_pca.joblib")
            joblib.dump(self.pca_model, pca_path, compress=3)
            logger.info(f"Saved fitted PCA to {pca_path}")

    def predict(self, X):
        """Return predictions in the ORIGINAL (~60k-d) target space."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y_hat = self.model.predict(X)  # in PCA space if use_pca else original space
        if self.use_pca:
            if self.pca_model is None:
                raise RuntimeError("PCA model missing despite use_pca=True.")
            y_hat = self.pca_model.inverse_transform(y_hat)
        return y_hat


class MLPRegression(RegressionModel):
    """MLP regression model with optional PCA preprocessing."""

    def __init__(self, target, roi, subject, fmri_dir, feat_dir, models_dir,
                 betas_flag, use_pca, pca_dim=1024,
                 hidden_units=1024, activation='relu',
                 max_iter=200, random_state=42):
        super().__init__(target, roi, subject, fmri_dir, feat_dir, models_dir, betas_flag, use_pca)
        self.pca_dim = pca_dim
        self.pca_model = None
        self.hidden_units = hidden_units
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        logger.info(f"Fitting MLPRegression for target={self.target}, ROIs={self.roi}, subject={self.subject}")
        mlp = MLPRegressor(
            hidden_layer_sizes=(self.hidden_units,),
            activation=self.activation,
            solver='adam',
            max_iter=self.max_iter,
            early_stopping=True,
            verbose=True,
            random_state=self.random_state
        )
        scaler = StandardScaler(with_mean=True, with_std=True)

        if self.use_pca:
            self.pca_model = PCA(n_components=self.pca_dim)
            self.model = make_pipeline(scaler, self.pca_model, mlp)
        else:
            self.model = make_pipeline(scaler, mlp)

        t0 = perf_counter()
        self.model.fit(X, y)
        logger.info(f"MLP fit done in {perf_counter() - t0:.2f}s")

        roi_str = "_".join(self.roi)
        pca_tag = f"pca{self.pca_dim}" if self.use_pca else "nopca"
        model_name = f"mlp_regressor_{pca_tag}_{self.betas_flag}_{self.target}_{roi_str}_{self.subject}.joblib"
        model_path = os.path.join(self.models_dir, model_name)
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.model, model_path, compress=3)
        self._saved_model_path = model_path
        logger.info(f"Saved full model pipeline to {model_path}")


class RidgeRegression(RegressionModel):
    """Ridge regression model using himalaya.RidgeCV."""

    def __init__(self, target, roi, subject, fmri_dir, feat_dir, models_dir, betas_flag, use_pca, pca_dim=1024):
        super().__init__(target, roi, subject, fmri_dir, feat_dir, models_dir, betas_flag, use_pca)
        self.pca_dim = pca_dim
        self.pca_model = None

    def fit(self, X, y):
        logger.info(f"Fitting RidgeRegression for target={self.target}, ROIs={self.roi}, subject={self.subject}")
        alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        ridge = RidgeCV(alphas=alphas)
        scaler = StandardScaler(with_mean=True, with_std=True)

        if self.use_pca:
            self.pca_model = PCA(n_components=self.pca_dim)
            self.model = make_pipeline(scaler, self.pca_model, ridge)
        else:
            self.model = make_pipeline(scaler, ridge)

        t0 = perf_counter()
        self.model.fit(X, y)
        logger.info(f"Ridge fit done in {perf_counter() - t0:.2f}s")

        # Save with model-specific filename
        roi_str = "_".join(self.roi)
        pca_tag = f"pca{self.pca_dim}" if self.use_pca else "nopca"
        model_name = f"ridge_regressor_{pca_tag}_{self.betas_flag}_{self.target}_{roi_str}_{self.subject}.joblib"
        model_path = os.path.join(self.models_dir, model_name)
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.model, model_path, compress=3)
        self._saved_model_path = model_path
        logger.info(f"Saved full model pipeline to {model_path}")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Train regression models for fMRI data.")
    parser.add_argument("--target", type=str, required=True, help="Feature type (init_latent or c_topk_captions)")
    parser.add_argument("--roi", required=True, type=str, nargs="*", help="ROI names (space-separated)")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., subj01)")
    parser.add_argument("--fmri_dir", type=str, default="/mnt/outputs/fmri_features", help="fMRI Betas data directory")
    parser.add_argument("--feat_dir", type=str, default="/mnt/outputs/latent_features",
                        help="Latent Features directory")
    parser.add_argument("--models_dir", type=str, default="/mnt/outputs/saved_models",
                        help="Output directory for models and predictions")
    parser.add_argument("--model", type=str, default="all", choices=['ridge', 'kernelridge', 'mlp', 'all'],
                        help="Model to train (default: all)")
    parser.add_argument("--pca_dim", type=int, default=1024, help="PCA components (default: 512)")
    parser.add_argument("--use_pca", action="store_true", help="Enable PCA in the pipeline")
    parser.add_argument("--betas_flag", type=str, default="each", choices=["each", "ave"],
                        help="Training betas granularity for X (each/ave)")
    parser.add_argument("--ridge_kernel", type=str, default="rbf",
                        choices=["rbf", "linear", "poly", "laplacian", "sigmoid", "cosine"],
                        help="Kernel for KernelRidge.")

    args = parser.parse_args()

    # safely get log_level from args with default "INFO"
    log_level = getattr(args, "log_level", "INFO")

    # set level
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Constructors with fixed kwargs via partial (so ** works when instantiating)
    models = {
        'ridge': RidgeRegression,
        'kernelridge': partial(KernelRidgeRegression, ridge_kernel=args.ridge_kernel, pca_dim=args.pca_dim),
        'mlp': partial(MLPRegression, pca_dim=args.pca_dim)
    }

    selected_items = list(models.items()) if args.model == 'all' else [(args.model, models[args.model])]

    for model_name, ctor in tqdm(selected_items, desc="Training models", unit="model"):
        logger.info(f"▶ Starting {model_name} | subject={args.subject} | roi={args.roi} | target={args.target} | "
                    f"use_pca={args.use_pca} | pca_dim={args.pca_dim} | betas_flag={args.betas_flag}")
        reg_model = ctor(
            target=args.target,
            roi=args.roi,
            subject=args.subject,
            fmri_dir=args.fmri_dir,
            feat_dir=args.feat_dir,
            models_dir=args.models_dir,
            betas_flag=args.betas_flag,
            use_pca=args.use_pca
        )
        model_path, scores, mean_rs = reg_model.train()
        logger.info(f"✔ Finished {reg_model.__class__.__name__} → {model_path} (mean corr={mean_rs:.3f})")


if __name__ == "__main__":
    main()
