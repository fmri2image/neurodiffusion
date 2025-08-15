import numpy as np
import pickle
import os
import argparse
import logging
from abc import ABC, abstractmethod
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RegressionModel")


class RegressionModel(ABC):
    """Base class for regression models mapping fMRI to stimuli features."""

    def __init__(self, target, roi, subject, fmri_dir, feat_dir, models_dir):
        """
        Initialize the regression model.

        Args:
            target (str): Feature type (e.g., 'init_latent', 'c_top5_captions').
            roi (list): List of regions of interest (e.g., ['early', 'ventral']).
            subject (str): Subject ID (e.g., 'subj01').
            fmri_dir (str): Directory containing fMRI data.
            feat_dir (str): Directory containing train/test latent feature splits.
            models_dir (str): Directory to save trained models and predictions.
        """
        self.target = target
        self.roi = roi if isinstance(roi, list) else [roi]
        self.subject = subject
        self.fmri_dir = os.path.join(fmri_dir, subject)
        self.feat_dir = os.path.join(feat_dir, subject)
        self.models_dir = os.path.join(models_dir, subject)
        self.model = None
        self.backend = set_backend("torch_cuda" if 'mlp' in self.__class__.__name__.lower() else "numpy",
                                   on_error="warn")

    def load_data(self):
        """Load fMRI and feature data for training and testing."""
        try:
            # Load train/test indices
            tr_idx_path = os.path.join(self.fmri_dir, f"{self.subject}_stims_tridx.npy")
            tr_idx = np.load(tr_idx_path)

            # Load fMRI data for each ROI
            X, X_te = [], []
            for croi in self.roi:
                X.append(np.load(os.path.join(self.fmri_dir, f"{self.subject}_{croi}_betas_tr.npy")).astype("float32"))
                X_te.append(
                    np.load(os.path.join(self.fmri_dir, f"{self.subject}_{croi}_betas_ave_te.npy")).astype("float32"))
            X = np.hstack(X)
            X_te = np.hstack(X_te)

            # Load train/test features
            feat_tr_path = os.path.join(self.feat_dir, f"splits/{self.subject}_each_{self.target}_tr.npy")
            feat_te_path = os.path.join(self.feat_dir, f"splits/{self.subject}_ave_{self.target}_te.npy")
            Y = np.load(feat_tr_path).astype("float32").reshape(X.shape[0], -1)
            Y_te = np.load(feat_te_path).astype("float32").reshape(X_te.shape[0], -1)

            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"Mismatch between fMRI ({X.shape[0]}) and features ({Y.shape[0]}) samples.")

            logger.info(f"X shape: {X.shape}, Y shape: {Y.shape}, X_te shape: {X_te.shape}, Y_te shape: {Y_te.shape}")
            return X, Y, X_te, Y_te
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    @abstractmethod
    def fit(self, X, y):
        """Fit the regression model. Must be implemented by subclasses."""
        pass

    def save_model(self):
        """Save the trained model to models_dir."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        model_name = f"{self.__class__.__name__.lower()}_{self.target}_{'_'.join(self.roi)}_{self.subject}.pkl"
        os.makedirs(self.models_dir, exist_ok=True)
        output_path = os.path.join(self.models_dir, model_name)
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved model to {output_path}")
        return output_path

    def predict_and_evaluate(self, X_te, Y_te):
        """Predict on test data and compute correlation score."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        scores = self.model.predict(X_te)
        rs = correlation_score(Y_te.T, scores.T)
        mean_rs = np.mean(rs.cpu().numpy() if hasattr(rs, 'cpu') else rs)
        logger.info(f"Prediction correlation score: {mean_rs:.3f}")

        # Save predicted scores
        scores_path = os.path.join(self.feat_dir,
                                   f"scores/{self.subject}_{'_'.join(self.roi)}_scores_{self.target}.npy")
        np.save(scores_path, scores)
        logger.info(f"Saved predicted scores to {scores_path}")

        return scores, mean_rs

    def train(self):
        """Train the model, save it, and evaluate on test data."""
        X, Y, X_te, Y_te = self.load_data()
        self.fit(X, Y)
        model_path = self.save_model()
        scores, mean_rs = self.predict_and_evaluate(X_te, Y_te)
        return model_path, scores, mean_rs


class RidgeRegression(RegressionModel):
    """Ridge regression model using himalaya.RidgeCV."""

    def fit(self, X, y):
        """Fit the ridge regression model with cross-validated alphas."""
        logger.info(f"Fitting RidgeRegression for target {self.target}, ROIs {self.roi}, subject {self.subject}")
        alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1] if self.target in ['init_latent',
                                                                                     'c_top5_captions'] else [10000,
                                                                                                              20000,
                                                                                                              40000]
        ridge = RidgeCV(alphas=alphas)
        self.model = make_pipeline(StandardScaler(with_mean=True, with_std=True), ridge)
        self.model.fit(X, y)


class KernelRidgeRegression(RegressionModel):
    """Kernel-based ridge regression model for non-linear fMRI-to-feature mapping."""

    def __init__(self, target, roi, subject, fmri_dir, feat_dir, models_dir, pca_dim=512):
        super().__init__(target, roi, subject, fmri_dir, feat_dir, models_dir)
        self.pca_dim = pca_dim
        self.pca_model = None

    def fit(self, X, y):
        """Fit the kernel ridge regression model with RBF kernel and PCA."""
        logger.info(
            f"Fitting KernelRidgeRegression for target {self.target}, ROIs {self.roi}, subject {self.subject} with PCA={self.pca_dim}")
        self.pca_model = PCA(n_components=self.pca_dim)
        kernel_ridge = KernelRidge(kernel='rbf', alpha=1e-3, gamma=None)
        self.model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            kernel_ridge
        )
        self.model.fit(X, y)

        # Save PCA model
        pca_path = os.path.join(self.models_dir, 'pca_models', f"pca_model_{self.target}_{self.subject}.joblib")
        os.makedirs(os.path.dirname(pca_path), exist_ok=True)
        joblib.dump(self.pca_model, pca_path)
        logger.info(f"Saved PCA model to {pca_path}")


class MLPRegression(RegressionModel):
    """MLP regression model with PCA preprocessing."""

    def __init__(self, target, roi, subject, fmri_dir, feat_dir, models_dir, pca_dim=512):
        """
        Initialize MLP regression with PCA parameters.

        Args:
            pca_dim (int): Number of PCA components.
            hidden_dim (int): Size of hidden layer in MLP.
        """
        super().__init__(target, roi, subject, fmri_dir, feat_dir, models_dir)
        self.pca_dim = pca_dim
        self.pca_model = None

    def fit(self, X, y):
        """Fit the MLP regression model with PCA preprocessing."""
        logger.info(f"Fitting MLPRegression for target {self.target}, ROIs {self.roi}, subject {self.subject}")
        self.pca_model = PCA(n_components=self.pca_dim)
        mlp = MLPRegressor(hidden_layer_sizes=(1024,), activation='relu', solver='adam',
                           max_iter=200, early_stopping=True, verbose=True, random_state=42)
        self.model = make_pipeline(StandardScaler(), self.pca_model, mlp)
        self.model.fit(X, y)

        # Save PCA model
        pca_path = os.path.join(self.models_dir, 'pca_models', f"pca_model_{self.target}_{self.subject}.joblib")
        os.makedirs(os.path.dirname(pca_path), exist_ok=True)
        joblib.dump(self.pca_model, pca_path)
        logger.info(f"Saved PCA model to {pca_path}")


def main():
    parser = argparse.ArgumentParser(description="Train regression models for fMRI data.")
    parser.add_argument("--target", type=str, required=True, help="Feature type (init_latent or c_topk_captions)")
    parser.add_argument("--roi", type=str, required=True, nargs='+',
                        help="Regions of interest (e.g., early ventral midventral ...)")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., subj01)")
    parser.add_argument("--fmri_dir", type=str, default="/mnt/outputs/fmri_features", help="fMRI Betas data directory")
    parser.add_argument("--latent_feat_dir", type=str, default="/mnt/outputs/latent_features",
                        help="Latent Features directory")
    parser.add_argument("--models_dir", type=str, default="/mnt/outputs/saved_models",
                        help="Output directory for models and predictions")
    parser.add_argument("--model", type=str, default="all", choices=['ridge', 'kernelridge', 'mlp', 'all'],
                        help="Model to train (default: all)")
    parser.add_argument("--pca_dim", type=int, default=512, help="PCA components for ant model (default: 512)")

    args = parser.parse_args()

    models = {
        'ridge': RidgeRegression,
        'kernelridge': lambda *x: KernelRidgeRegression(*x, pca_dim=args.pca_dim),
        'mlp': lambda *x: MLPRegression(*x, pca_dim=args.pca_dim)
    }

    if args.model == 'all':
        selected_models = models.values()
    else:
        selected_models = [models[args.model]]

    for model_class in selected_models:
        model = model_class(
            target=args.target,
            roi=args.roi,
            subject=args.subject,
            fmri_dir=args.fmri_dir,
            feat_dir=args.latent_feat_dir,
            models_dir=args.models_dir
        )
        model_path, scores, mean_rs = model.train()
        print(
            f"{model_class.__name__ if hasattr(model_class, '__name__') else 'MLPRegression'} saved to: {model_path}, Mean correlation: {mean_rs:.3f}")


if __name__ == "__main__":
    main()