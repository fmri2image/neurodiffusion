import numpy as np
import torch
import os
import argparse
import logging
import h5py
from PIL import Image
import scipy.io
import joblib
from omegaconf import OmegaConf
from tqdm import trange
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import PIL

from nsd_access.nsda import NSDAccess
import sys;

sys.path.append("/mnt/packages/stable-diffusion")
sys.path.append("/mnt/packages/taming-transformers")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiffusionDecoder")


class DiffusionDecoder:
    """Class to reconstruct images from predicted latent features using Stable Diffusion."""

    def __init__(self, imgidx, method, subject, gpu, seed, nsd_dir, output_dir, config_path, ckpt_path):
        """
        Initialize the diffusion decoder.

        Args:
            imgidx (list): Single index or range of image indices.
            method (str): Decoding method ('ridge', 'kernel-ridge', 'mlp-regressor').
            subject (str): Subject ID (e.g., 'subj01').
            gpu (int): GPU index.
            seed (int): Random seed for reproducibility.
            nsd_dir (str): Directory containing NSD data.
            output_dir (str): Directory to save reconstructed images.
            config_path (str): Path to Stable Diffusion config.
            ckpt_path (str): Path to Stable Diffusion checkpoint.
        """
        self.imgidx = [int(x.strip()) for x in imgidx.split(",")]
        self.method = method
        self.subject = subject
        self.gpu = gpu
        self.seed = seed
        self.nsd_dir = nsd_dir
        self.output_dir = output_dir
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")

        # Validate method
        if self.method not in ['ridge', 'kernel-ridge', 'mlp-regressor']:
            raise ValueError(f"Method must be one of: ridge, kernel-ridge, mlp-regressor. Got: {method}")

        # Set parameters
        self.n_samples = 1
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.strength = 0.8
        self.scale = 5.0
        self.n_iter = 5
        self.batch_size = self.n_samples
        self.precision = 'autocast'

        # Initialize NSD access and model
        seed_everything(self.seed)
        self.load_model()

        # Load NSD experiment design
        self.nsd_expdesign = scipy.io.loadmat(
            os.path.join(self.nsd_dir, 'nsddata', 'experiments', 'nsd', 'nsd_expdesign.mat'))
        self.sharedix = self.nsd_expdesign['sharedix'][0] - 1  # Convert to 0-based indexing

        # Load stimuli indices
        stim_path = os.path.join(self.output_dir, f'fmri_features/{self.subject}/{self.subject}_stims_ave.npy')
        self.stims_ave = np.load(stim_path)
        self.tr_idx = np.zeros_like(self.stims_ave)
        for idx, s in enumerate(self.stims_ave):
            self.tr_idx[idx] = 0 if s in self.sharedix else 1

        # Set output paths
        self.sample_path = os.path.join(self.output_dir, f"reconstructed_images/{method}", subject)
        os.makedirs(self.sample_path, exist_ok=True)

    def load_model(self):
        """Load Stable Diffusion model."""
        logger.info(f"Loading model from {self.ckpt_path}")
        config = OmegaConf.load(self.config_path)
        pl_sd = torch.load(self.ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"]
        self.model = instantiate_from_config(config.model)
        m, u = self.model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logger.warning(f"Missing keys: {m}")
        if len(u) > 0:
            logger.warning(f"Unexpected keys: {u}")
        self.model.to(self.device).eval()
        self.sampler = DDIMSampler(self.model)
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

    def load_img_from_arr(self, img_arr):
        """Convert image array to latent input."""
        image = Image.fromarray(img_arr).convert("RGB")
        image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(self.device)
        return 2. * image - 1.

    def _load_init_latent(self, imgidx):
        """Load predicted init_latent with PCA inverse transform if needed."""
        roi_latent = 'early'
        if self.method in ['mlp-regressor']:
            scores_path = os.path.join(
                self.output_dir,
                os.path.join(self.output_dir,
                             f"latent_features/{self.subject}/scores/{self.subject}_early_scores_init_latent.npy")
            )
            scores_latent = np.load(scores_path)
            z_reduced = scores_latent[imgidx].reshape(1, -1)

            # Load corresponding PCA model and inverse transform
            pca_path = os.path.join(
                self.output_dir,
                f"{self.method}/pca_models/pca_model_init_latent_{self.subject}.joblib"
            )
            pca = joblib.load(pca_path)
            z = pca.inverse_transform(z_reduced)[0]  # shape should be (6400,)
        else:
            # No PCA
            scores_path = os.path.join(self.output_dir,
                                       f"latent_features/{self.subject}/scores/{self.subject}_early_scores_init_latent.npy")
            scores_latent = np.load(scores_path)
            z = scores_latent[imgidx]

        if z.shape[0] != 6400:
            raise ValueError(f"Expected init_latent shape (6400,), got {z.shape[0]}")

        imgarr = torch.tensor(z.reshape(1, 4, 40, 40), dtype=torch.float32).to(self.device)
        return imgarr

    def _load_conditioning(self, imgidx):
        """Load predicted conditioning embedding, with PCA inverse transform if needed."""
        roi_c = 'ventral'

        if self.method in ['mlp-regressor', 'kernel-ridge']:
            # Correct paths for scores and PCA model
            scores_path = os.path.join(
                self.output_dir,
                f"latent_features/{self.subject}/scores/{self.subject}_{roi_c}_scores_c_top1_captions.npy"
            )
            pca_path = os.path.join(
                self.output_dir,
                f"models/{self.subject}/pca_models/pca_model_c_top1_captions_{self.subject}.joblib"
            )

            # Load and inverse PCA
            scores_c = np.load(scores_path)
            c_full = scores_c[imgidx].reshape(1, 77, 768)

            # c_compressed = scores_c[imgidx].reshape(1, -1)
            # pca = joblib.load(pca_path)

            # print("scores_c.shape:", scores_c.shape)
            # print("PCA n_components_:", pca.n_components_)
            # print("c_compressed: ", c_compressed.shape)

            # ivc = pca.inverse_transform(c_compressed)

            # print(ivc.shape)

            # c_full = ivc.reshape(1, 77, 768)

        else:
            scores_path = os.path.join(self.output_dir, f"{self.subject}_{roi_c}_scores_c_top1_captions.npy")
            c_full = np.load(scores_path)[imgidx].reshape(1, 77, 768)

        return torch.tensor(c_full, dtype=torch.float32).to(self.device)

    def decode(self):
        """Reconstruct images for the specified indices and method."""
        # Handle image index input
        if len(self.imgidx) == 1:
            imgidx_list = [self.imgidx[0]]
        elif len(self.imgidx) == 2:
            imgidx_list = list(range(self.imgidx[0], self.imgidx[1] + 1))
        else:
            raise ValueError("Please provide either one index or a range of two indices (start end).")

        precision_scope = autocast if self.precision == "autocast" else nullcontext

        for imgidx in imgidx_list:
            logger.info(f"Decoding image {imgidx:05} with method {self.method}")

            # Save original image
            imgidx_te = np.where(self.tr_idx == 0)[0][imgidx]  # Map to test index
            idx73k = self.stims_ave[imgidx_te]

            nsda = NSDAccess(self.nsd_dir)
            sf = h5py.File(nsda.stimuli_file, 'r')
            sdataset = sf.get('imgBrick')

            Image.fromarray(np.squeeze(sdataset[idx73k, :, :, :]).astype(np.uint8)).save(
                os.path.join(self.sample_path, f"{imgidx:05}_org.png"))

            # Load and decode init_latent
            imgarr = self._load_init_latent(imgidx)
            with torch.no_grad(), precision_scope("cuda"):
                with self.model.ema_scope():
                    x_samples = self.model.decode_first_stage(imgarr)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
            im = Image.fromarray(x_sample.astype(np.uint8)).resize((512, 512))
            im = np.array(im)

            # Encode to latent space
            init_image = self.load_img_from_arr(im)
            init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))

            # Load conditioning
            c = self._load_conditioning(imgidx)

            # Generate reconstructed images
            with torch.no_grad(), precision_scope("cuda"):
                with self.model.ema_scope():
                    for n in trange(self.n_iter, desc="Sampling"):
                        uc = self.model.get_learned_conditioning(self.batch_size * [""])
                        z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor(
                            [int(self.strength * self.ddim_steps)] * self.batch_size).to(self.device))
                        samples = self.sampler.decode(z_enc, c, int(self.strength * self.ddim_steps),
                                                      unconditional_guidance_scale=self.scale,
                                                      unconditional_conditioning=uc)
                        x_samples = self.model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        for idx, x_sample in enumerate(x_samples):
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(self.sample_path, f"{imgidx:05}_{idx:03}.png"))


def main():
    parser = argparse.ArgumentParser(description="Reconstruct images from predicted latent features.")
    parser.add_argument("--imgidx", type=int, required=True, nargs='+', help="Image index or range (start end)")
    parser.add_argument("--method", type=str, required=True, choices=['ridge', 'kernel-ridge', 'mlp-regressor'],
                        help="Decoding method")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., subj01)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nsd_dir", type=str, default="data/processed/nsd", help="NSD data directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--config_path", type=str,
                        default="diffusion_sd1/stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
                        help="Stable Diffusion config path")
    parser.add_argument("--ckpt_path", type=str, default="data/models/stable-diffusion/sd-v1-4.ckpt",
                        help="Stable Diffusion checkpoint path")

    args = parser.parse_args()

    decoder = DiffusionDecoder(
        imgidx=args.imgidx,
        method=args.method,
        subject=args.subject,
        gpu=args.gpu,
        seed=args.seed,
        nsd_dir=args.nsd_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        ckpt_path=args.ckpt_path
    )
    decoder.decode()


if __name__ == "__main__":
    main()