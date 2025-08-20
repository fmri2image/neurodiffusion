import argparse
import os
import numpy as np
import h5py
import scipy.io
import torch
from torch import autocast
from contextlib import nullcontext
from PIL import Image
import PIL
from tqdm import trange
from einops import rearrange
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
import sys

sys.path.append("/mnt/packages/stable-diffusion")
sys.path.append("/mnt/packages/taming-transformers")

from nsd_access.nsda import NSDAccess
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class DiffusionDecoder:
    """Reconstruct images from predicted latent features (assumes full-size predictions)."""

    def __init__(
        self,
        imgidx: int,
        method: str,
        subject: str,
        gpu: int,
        nsd_dir: str,
        output_dir: str,
        packages_path: str,
        roi_init_latent,
        roi_c,
        captions_type: str,
        kernel_used: str,
        pca_dim: str,
        seed: int = 42,
        ddim_steps: int = 50,
        strength: float = 0.8,
        scale: float = 5.0,
        n_iter: int = 5,
    ):
        self.imgidx = int(imgidx)
        self.method = method
        self.subject = subject
        self.gpu = gpu
        self.nsd_dir = nsd_dir.rstrip("/")
        self.output_dir = output_dir.rstrip("/")
        self.fmri_dir = os.path.join(self.output_dir, "fmri_features")
        self.packages_path = packages_path.rstrip("/")
        self.sd_config = os.path.join(self.packages_path,
                                      "stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
        self.sd_ckpt = os.path.join(self.packages_path,
                                    "stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt")

        # Device & AMP
        self.device = torch.device(f"cuda:{self.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        self._amp = (lambda: autocast(device_type="cuda")) if torch.cuda.is_available() else (lambda: nullcontext())

        # ROIs (accept single or list)
        self.roi_init_latent = roi_init_latent if isinstance(roi_init_latent, list) else [roi_init_latent]
        self.roi_c = roi_c if isinstance(roi_c, list) else [roi_c]
        self.roi_str = "_".join(self.roi_init_latent)
        self.roi_c_str = "_".join(self.roi_c)

        self.captions_type = captions_type
        self.kernel = kernel_used
        self.pca_dim = pca_dim  # only used in naming (since predictions are full-size)

        # Sampling
        self.n_samples = 1
        self.ddim_steps = int(ddim_steps)
        self.ddim_eta = 0.0
        self.strength = float(strength)
        self.scale = float(scale)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.batch_size = self.n_samples
        assert 0.0 <= self.strength <= 1.0, "strength must be in [0, 1]"
        self.t_enc = int(self.strength * self.ddim_steps)

        # Setup
        seed_everything(self.seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)
            torch.backends.cudnn.benchmark = True

        self._load_sd_model()
        self._load_nsd_and_indices()

        # Clean output naming
        parts = [self.method]
        if "kernel" in self.method.lower():
            parts.append(self.kernel)
        if self.pca_dim:
            parts.append(self.pca_dim)
        parts += [self.roi_str, "init_latent", self.roi_c_str, self.captions_type]
        name = "_".join(parts)

        self.sample_path = os.path.join(
            self.output_dir, "reconstructed_images_from_predictions", name, self.subject
        )
        os.makedirs(self.sample_path, exist_ok=True)

    # ---------- SD model ----------
    def _load_sd_model(self):
        config = OmegaConf.load(self.sd_config)
        print(f"[INFO] Loading SD model from {self.sd_ckpt}")
        pl_sd = torch.load(self.sd_ckpt, map_location="cpu")
        sd = pl_sd["state_dict"] if isinstance(pl_sd, dict) and "state_dict" in pl_sd else pl_sd
        model = instantiate_from_config(config.model)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:   print("[WARN] Missing keys:", missing)
        if unexpected:print("[WARN] Unexpected keys:", unexpected)
        self.model = model.to(self.device).eval()
        self.sampler = DDIMSampler(self.model)
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

    # ---------- NSD / indices ----------
    def _load_nsd_and_indices(self):
        exp_path = os.path.join(self.nsd_dir, "nsddata/experiments/nsd/nsd_expdesign.mat")
        if not os.path.exists(exp_path):
            raise FileNotFoundError(f"NSD expdesign not found: {exp_path}")
        nsd_expdesign = scipy.io.loadmat(exp_path)
        sharedix = np.array(nsd_expdesign["sharedix"]).ravel() - 1  # flatten
        shared_set = set(sharedix.tolist())

        self.nsda = NSDAccess(self.nsd_dir)
        self.h5 = h5py.File(self.nsda.stimuli_file, "r")
        self.sdataset = self.h5["imgBrick"]

        stims_path = os.path.join(self.fmri_dir, self.subject, f"{self.subject}_stims_ave.npy")
        if not os.path.exists(stims_path):
            raise FileNotFoundError(f"stims_ave not found: {stims_path}")
        self.stims_ave = np.load(stims_path)

        self.tr_idx = np.zeros_like(self.stims_ave, dtype=np.int32)
        for i, s in enumerate(self.stims_ave):
            self.tr_idx[i] = 0 if int(s) in shared_set else 1

    # ---------- helpers ----------
    @staticmethod
    def _to_image_tensor(arr_uint8_hw3: np.ndarray, device: torch.device) -> torch.Tensor:
        img = Image.fromarray(arr_uint8_hw3).convert("RGB")
        img = img.resize((512, 512), resample=PIL.Image.LANCZOS)
        x = np.array(img).astype(np.float32) / 255.0
        x = x[None].transpose(0, 3, 1, 2)
        x = torch.from_numpy(x).to(device)
        return 2.0 * x - 1.0

    # ---------- z -> init image ----------
    def _load_z_image_tensor(self) -> torch.Tensor:
        # map legacy test index to 73k idx, save GT
        test_positions = np.where(self.tr_idx == 0)[0]
        if self.imgidx < 0 or self.imgidx >= len(test_positions):
            raise IndexError(f"imgidx {self.imgidx} out of test-set range (0..{len(test_positions)-1})")
        imgidx_te = test_positions[self.imgidx]
        idx73k = int(self.stims_ave[imgidx_te])
        gt = np.squeeze(self.sdataset[idx73k, :, :, :]).astype(np.uint8)
        Image.fromarray(gt).save(os.path.join(self.sample_path, f"{self.imgidx:05}_org.png"))

        if self.method == 'kernelridgeregression':
            z_scores_path = os.path.join(
                self.output_dir,
                f"latent_features/splitted_latents/{self.subject}/prediction_scores/{self.method}/{self.kernel}_kernel/"
                f"{self.subject}_{self.roi_str}_scores_init_latent.npy"
            )
        else:
            z_scores_path = os.path.join(
                self.output_dir,
                f"latent_features/splitted_latents/{self.subject}/prediction_scores/{self.method}/"
                f"{self.subject}_{self.roi_str}_scores_init_latent.npy"
            )

        if not os.path.exists(z_scores_path):
            raise FileNotFoundError(f"z scores not found: {z_scores_path}")
        z_all = np.load(z_scores_path)              # [Ntest, 6400]
        z_vec = z_all[self.imgidx, :]
        if z_vec.shape[-1] != 6400:
            raise ValueError(f"Expected init_latent 6400 dims, got {z_vec.shape[-1]}")
        imgarr = torch.tensor(z_vec.reshape(1, 4, 40, 40), dtype=torch.float32, device=self.device)

        # decode predicted latent to RGB, then re-encode as init image (matches your pipeline)
        with torch.no_grad(), self._amp():
            with self.model.ema_scope():
                x_samples = self.model.decode_first_stage(imgarr)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
                x = 255.0 * rearrange(x_samples[0].cpu().numpy(), "c h w -> h w c")
        return self._to_image_tensor(x.astype(np.uint8), self.device)

    # ---------- conditioning c ----------
    def _load_conditioning(self) -> torch.Tensor:
        if self.method == 'kernelridgeregression':
            c_scores_path = os.path.join(
                self.output_dir,
                f"latent_features/splitted_latents/{self.subject}/prediction_scores/{self.method}/{self.kernel}_kernel/"
                f"{self.subject}_{self.roi_c_str}_scores_{self.captions_type}.npy"
            )
        else:
            c_scores_path = os.path.join(
                self.output_dir,
                f"latent_features/splitted_latents/{self.subject}/prediction_scores/{self.method}/"
                f"{self.subject}_{self.roi_c_str}_scores_{self.captions_type}.npy"
            )
        if not os.path.exists(c_scores_path):
            raise FileNotFoundError(f"c scores not found: {c_scores_path}")
        c_all = np.load(c_scores_path)  # [Ntest, 59136] or [Ntest, 77, 768]
        c_item = c_all[self.imgidx]
        if c_item.ndim == 1:
            if c_item.shape[0] != 77 * 768:
                raise ValueError(f"Expected 59136 dims for c, got {c_item.shape[0]}")
            c_77x768 = c_item.reshape(1, 77, 768)
        elif c_item.ndim == 2 and c_item.shape == (77, 768):
            c_77x768 = c_item.reshape(1, 77, 768)
        elif c_item.ndim == 3 and c_item.shape == (1, 77, 768):
            c_77x768 = c_item
        else:
            raise ValueError(f"Unexpected c shape: {c_item.shape}")
        return torch.tensor(c_77x768, dtype=torch.float32, device=self.device)

    # ---------- decode ----------
    def decode(self):
        print(f"target t_enc is {self.t_enc} steps")
        init_image = self._load_z_image_tensor()
        with torch.no_grad():
            init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))

        c = self._load_conditioning()

        base_count = 0
        with torch.no_grad(), self._amp():
            with self.model.ema_scope():
                for _ in trange(self.n_iter, desc="Sampling"):
                    uc = self.model.get_learned_conditioning(self.batch_size * [""])
                    z_enc = self.sampler.stochastic_encode(
                        init_latent, torch.tensor([self.t_enc] * self.batch_size, device=self.device)
                    )
                    samples = self.sampler.decode(
                        z_enc, c, self.t_enc,
                        unconditional_guidance_scale=self.scale,
                        unconditional_conditioning=uc,
                    )
                    x_samples = self.model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
                    for x_sample in x_samples:
                        x_np = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                        Image.fromarray(x_np.astype(np.uint8)).save(
                            os.path.join(self.sample_path, f"{self.imgidx:05}_{base_count:03}.png")
                        )
                        base_count += 1

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Decode images from predicted z/c (full-size predictions).")
    parser.add_argument("--imgidx", type=int, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nsd_dir", type=str, default="/mnt/datasets/nsd")
    parser.add_argument("--output_dir", type=str, default="/mnt/outputs")
    parser.add_argument("--packages_dir", type=str, default="/mnt/packages")
    parser.add_argument("--roi_init_latent", type=str, nargs="+", required=True)
    parser.add_argument("--roi_c", type=str, nargs="+", required=True)
    parser.add_argument("--captions_type", type=str, required=True)      # e.g., c_top1_captions
    parser.add_argument("--kernel_used", type=str, default="poly")
    parser.add_argument("--pca_dim", type=str, default="no-pca")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--n_iter", type=int, default=5)
    a = parser.parse_args()

    dec = DiffusionDecoder(
        imgidx=a.imgidx,
        method=a.method,
        subject=a.subject,
        gpu=a.gpu,
        nsd_dir=a.nsd_dir,
        output_dir=a.output_dir,          # <- IMPORTANT: pass output_dir
        packages_path=a.packages_dir,
        roi_init_latent=a.roi_init_latent,
        roi_c=a.roi_c,
        captions_type=a.captions_type,
        kernel_used=a.kernel_used,
        pca_dim=a.pca_dim,
        seed=a.seed,
        ddim_steps=a.ddim_steps,
        strength=a.strength,
        scale=a.scale,
        n_iter=a.n_iter,
    )
    try:
        dec.decode()
    finally:
        dec.close()


if __name__ == "__main__":
    main()
