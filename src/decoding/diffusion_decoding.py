import argparse
import os
import numpy as np
import h5py
import scipy.io
import torch
from PIL import Image
import PIL
from tqdm import trange, tqdm
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from contextlib import nullcontext

import sys
sys.path.append("/mnt/packages/stable-diffusion")
sys.path.append("/mnt/packages/taming-transformers")

from nsd_access.nsda import NSDAccess
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def _enable_speedups():
    if torch.cuda.is_available():
        # Allow TF32 (Ampere+) for matmuls; nice speedup with imperceptible quality change
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


class DiffusionDecoder:
    """Reconstruct images from predicted latent features (assumes full-size predictions)."""

    def __init__(
        self,
        # core params
        method: str,
        subject: str,
        gpu: int,
        nsd_dir: str,
        output_dir: str,
        packages_path: str,
        roi_init_latent,
        roi_c,
        captions_type: str,
        kernel_used_c: str,
        kernel_used_init_latent: str,
        pca_dim_c: str,
        pca_dim_init_latent: str,
        betas_flag_c: str,
        betas_flag_init_latent: str,
        # sampling params
        seed: int = 42,
        ddim_steps: int = 50,
        strength: float = 0.8,
        scale: float = 5.0,
        n_iter: int = 5,
        sample_bs: int = 1,         # batch multiple variations per pass
        # opt flags (safe defaults)
        try_xformers: bool = True,  # attempt xFormers if available
        try_compile: bool = True,   # torch.compile if PyTorch 2+
    ):
        self.method = method
        self.subject = subject
        self.gpu = gpu
        self.nsd_dir = nsd_dir.rstrip("/")
        self.output_dir = output_dir.rstrip("/")
        self.fmri_dir = os.path.join(self.output_dir, "fmri_features")
        self.packages_path = packages_path.rstrip("/")
        self.betas_flag_c = betas_flag_c
        self.betas_flag_init_latent = betas_flag_init_latent

        self.sd_config = os.path.join(self.packages_path,
                                      "stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
        self.sd_ckpt = os.path.join(self.packages_path,
                                    "stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt")

        # Device & AMP
        self.device = torch.device(f"cuda:{self.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        self._amp = (lambda: torch.cuda.amp.autocast(dtype=torch.float16)) if torch.cuda.is_available() else (lambda: nullcontext())

        # ROIs (accept single or list)
        self.roi_init_latent = roi_init_latent if isinstance(roi_init_latent, list) else [roi_init_latent]
        self.roi_c = roi_c if isinstance(roi_c, list) else [roi_c]
        self.roi_str = "_".join(self.roi_init_latent)
        self.roi_c_str = "_".join(self.roi_c)

        self.captions_type = captions_type
        self.kernel_c = kernel_used_c
        self.kernel_init_latent = kernel_used_init_latent
        self.pca_dim_c = pca_dim_c              # naming only
        self.pca_dim_init_latent = pca_dim_init_latent  # naming only

        # Sampling
        self.ddim_steps = int(ddim_steps)
        self.ddim_eta = 0.0
        self.strength = float(strength)
        self.scale = float(scale)
        self.n_iter = int(n_iter)
        self.sample_bs = max(1, int(sample_bs))  # batched variations
        self.seed = int(seed)
        assert 0.0 <= self.strength <= 1.0, "strength must be in [0, 1]"
        self.t_enc = int(self.strength * self.ddim_steps)

        # Setup
        seed_everything(self.seed)
        _enable_speedups()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        self._load_sd_model(try_xformers=try_xformers, try_compile=try_compile)
        self._load_nsd_and_indices()

        # Clean output naming (compact, readable, and fixes missing `self.` on pca_dim_c)
        is_kernel   = "kernel" in self.method.lower()
        init_kernel = f"_kernel_{self.kernel_init_latent}" if is_kernel else ""
        c_kernel    = f"_kernel_{self.kernel_c}"            if is_kernel else ""

        name = (
            f"{self.method}_"
            f"init_latent_info -> (roi_{self.roi_str}_pca_{self.pca_dim_init_latent}{init_kernel})"
            f"__c_info -> (roi_{self.roi_c_str}_pca_{self.pca_dim_c}{c_kernel}_captions_{self.captions_type})"
        )

        self.sample_path = os.path.join(
            self.output_dir, "reconstructed_images_from_predictions", name, self.subject
        )
        os.makedirs(self.sample_path, exist_ok=True)

        # Preload (memory-map) prediction score arrays once
        self._z_all = self._load_z_scores_mmap()
        self._c_all = self._load_c_scores_mmap()

        # Cache uc for typical batch sizes to avoid recomputation
        self._uc_cache = {}

    # ---------- SD model ----------
    def _load_sd_model(self, try_xformers=True, try_compile=True):
        config = OmegaConf.load(self.sd_config)
        print(f"[INFO] Loading SD model from {self.sd_ckpt}")
        pl_sd = torch.load(self.sd_ckpt, map_location="cpu")
        sd = pl_sd["state_dict"] if isinstance(pl_sd, dict) and "state_dict" in pl_sd else pl_sd
        model = instantiate_from_config(config.model)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:   print("[WARN] Missing keys:", missing)
        if unexpected: print("[WARN] Unexpected keys:", unexpected)

        model = model.to(self.device).eval()

        # Optional speedups
        if try_xformers:
            try:
                # Different SD repos expose different helpers—try both
                if hasattr(model, "enable_xformers_memory_efficient_attention"):
                    model.enable_xformers_memory_efficient_attention()
                elif hasattr(model, "set_use_memory_efficient_attention_xformers"):
                    model.set_use_memory_efficient_attention_xformers(True)
                print("[INFO] xFormers memory-efficient attention enabled.")
            except Exception as e:
                print(f"[INFO] xFormers not enabled: {e}")

        if try_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                print("[INFO] torch.compile enabled.")
            except Exception as e:
                print(f"[INFO] torch.compile not enabled: {e}")

        self.model = model
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

        self.test_positions = np.where(np.isin(self.stims_ave.astype(int), list(shared_set)))[0]

    # ---------- prediction scores (memmap) ----------
    def _z_scores_path(self):
        if self.method == 'kernelridgeregression':
            return os.path.join(
                self.output_dir,
                f"latent_features/splitted_latents/{self.subject}/prediction_scores/{self.method}/{self.kernel_init_latent}_kernel/"
                f"{self.subject}_{self.roi_str}_scores_init_latent_{self.betas_flag_init_latent}_{self.pca_dim_init_latent}.npy"
            )
        else:
            return os.path.join(
                self.output_dir,
                f"latent_features/splitted_latents/{self.subject}/prediction_scores/{self.method}/"
                f"{self.subject}_{self.roi_str}_scores_init_latent.npy"
            )

    def _c_scores_path(self):
        if self.method == 'kernelridgeregression':
            return os.path.join(
                self.output_dir,
                f"latent_features/splitted_latents/{self.subject}/prediction_scores/{self.method}/{self.kernel_c}_kernel/"
                f"{self.subject}_{self.roi_c_str}_scores_{self.captions_type}_{self.betas_flag_c}_{self.pca_dim_c}.npy"
            )
        else:
            return os.path.join(
                self.output_dir,
                f"latent_features/splitted_latents/{self.subject}/prediction_scores/{self.method}/"
                f"{self.subject}_{self.roi_c_str}_scores_{self.captions_type}.npy"
            )

    def _load_z_scores_mmap(self):
        p = self._z_scores_path()
        if not os.path.exists(p):
            raise FileNotFoundError(f"z scores not found: {p}")
        return np.load(p, mmap_mode="r")  # [Ntest, 6400]

    def _load_c_scores_mmap(self):
        p = self._c_scores_path()
        if not os.path.exists(p):
            raise FileNotFoundError(f"c scores not found: {p}")
        return np.load(p, mmap_mode="r")  # [Ntest, ...]

    # ---------- helpers ----------
    def _index_to_73k(self, imgidx):
        if imgidx < 0 or imgidx >= len(self.test_positions):
            raise IndexError(f"imgidx {imgidx} out of test-set range (0..{len(self.test_positions) - 1})")
        pos = self.test_positions[imgidx]
        return int(self.stims_ave[pos])

    def _save_gt_png(self, imgidx, idx73k):
        gt = np.squeeze(self.sdataset[idx73k, :, :, :]).astype(np.uint8)
        Image.fromarray(gt).save(os.path.join(self.sample_path, f"{imgidx:05}_org.png"))

    def _get_uc(self, bs):
        if bs not in self._uc_cache:
            with torch.inference_mode(), self._amp():
                self._uc_cache[bs] = self.model.get_learned_conditioning(bs * [""])
        return self._uc_cache[bs]

    # ---------- build inputs (GPU) ----------
    def _build_init_image_and_c(self, imgidx):
        """Returns (init_image_tensor_in_-1..1, c_tensor) both on device."""
        # z → RGB (on GPU) → keep as [-1,1] latent image (avoid CPU roundtrip)
        z_vec = self._z_all[imgidx]  # shape (6400,)
        if z_vec.shape[-1] != 6400:
            raise ValueError(f"Expected init_latent 6400 dims, got {z_vec.shape[-1]}")
        z_latent = torch.as_tensor(z_vec, dtype=torch.float32, device=self.device).reshape(1, 4, 40, 40)

        with torch.inference_mode(), self._amp():
            with self.model.ema_scope():
                x_samples = self.model.decode_first_stage(z_latent)   # [-1,1]
                init_image = x_samples  # already in [-1,1]

        # conditioning c
        c_item = self._c_all[imgidx]
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

        c = torch.as_tensor(c_77x768, dtype=torch.float32, device=self.device)
        return init_image, c

    # ---------- decode single index ----------
    def decode_one(self, imgidx: int):
        print(f"target t_enc is {self.t_enc} steps")
        idx73k = self._index_to_73k(imgidx)
        self._save_gt_png(imgidx, idx73k)

        init_image, c = self._build_init_image_and_c(imgidx)
        with torch.inference_mode(), self._amp():
            init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))

        # sample in mini-batches for speed
        remaining = self.n_iter
        base_count = 0
        with torch.inference_mode(), self._amp():
            with self.model.ema_scope():
                while remaining > 0:
                    bs = min(self.sample_bs, remaining)
                    uc = self._get_uc(bs)
                    # tile latent & c to batch
                    z0 = init_latent.expand(bs, -1, -1, -1).contiguous()
                    cc = c.expand(bs, -1, -1).contiguous()

                    z_enc = self.sampler.stochastic_encode(
                        z0, torch.full((bs,), self.t_enc, device=self.device, dtype=torch.long)
                    )
                    samples = self.sampler.decode(
                        z_enc, cc, self.t_enc,
                        unconditional_guidance_scale=self.scale,
                        unconditional_conditioning=uc,
                    )
                    x_samples = self.model.decode_first_stage(samples)  # [-1,1]
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)

                    xs = x_samples.mul(255.0).permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
                    for i in range(xs.shape[0]):
                        Image.fromarray(xs[i]).save(
                            os.path.join(self.sample_path, f"{imgidx:05}_{base_count:03}.png")
                        )
                        base_count += 1
                    remaining -= bs

    # ---------- decode many ----------
    def decode_many(self, indices):
        for idx in tqdm(indices, desc="Decoding indices", unit="img"):
            self.decode_one(idx)

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass


def _normalize_indices(imgidx_list):
    if len(imgidx_list) == 1:
        return [imgidx_list[0]]
    elif len(imgidx_list) == 2:
        start, end = imgidx_list
        if start > end:
            start, end = end, start
        return list(range(start, end + 1))  # inclusive
    else:
        raise ValueError("Pass either one index or exactly two numbers for a range.")


def main():
    parser = argparse.ArgumentParser(description="Decode images from predicted z/c (full-size predictions).")
    parser.add_argument("--imgidx", nargs="+", type=int, required=True,
                        help="Either a single index (e.g., 7) or a range 'start end' (e.g., 1 10) inclusive.")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--roi_init_latent", type=str, nargs="+", required=True)
    parser.add_argument("--roi_c", type=str, nargs="+", required=True)
    parser.add_argument("--captions_type", type=str, required=True)  # e.g., c_top1_captions
    parser.add_argument("--kernel_used_c", type=str, default="poly")
    parser.add_argument("--kernel_used_init_latent", type=str, default="rbf")
    parser.add_argument("--pca_dim_c", type=str, default="nopca",
                        help="use 'nopca' or labels like 'pca512', 'pca1024', etc. (naming only)")
    parser.add_argument("--pca_dim_init_latent", type=str, default="nopca",
                        help="use 'nopca' or labels like 'pca512', 'pca1024', etc. (naming only)")
    parser.add_argument("--betas_flag_init_latent", type=str, default="each", choices=["each", "ave"])
    parser.add_argument("--betas_flag_c", type=str, default="each", choices=["each", "ave"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--n_iter", type=int, default=5)
    parser.add_argument("--sample_bs", type=int, default=1, help="Batch size of variations per image (speed).")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nsd_dir", type=str, default="/mnt/datasets/nsd")
    parser.add_argument("--output_dir", type=str, default="/mnt/outputs")
    parser.add_argument("--packages_dir", type=str, default="/mnt/packages")
    parser.add_argument("--no_xformers", action="store_true", help="Disable xFormers attempt.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile attempt.")

    a = parser.parse_args()
    indices = _normalize_indices(a.imgidx)

    decoder = DiffusionDecoder(
        method=a.method,
        subject=a.subject,
        gpu=a.gpu,
        nsd_dir=a.nsd_dir,
        output_dir=a.output_dir,
        packages_path=a.packages_dir,
        roi_init_latent=a.roi_init_latent,
        roi_c=a.roi_c,
        captions_type=a.captions_type,
        kernel_used_init_latent=a.kernel_used_init_latent,
        kernel_used_c=a.kernel_used_c,
        pca_dim_c=a.pca_dim_c,
        pca_dim_init_latent=a.pca_dim_init_latent,
        seed=a.seed,
        ddim_steps=a.ddim_steps,
        strength=a.strength,
        scale=a.scale,
        n_iter=a.n_iter,
        sample_bs=a.sample_bs,
        betas_flag_c=a.betas_flag_c,
        betas_flag_init_latent=a.betas_flag_init_latent,
        try_xformers=(not a.no_xformers),
        try_compile=(not a.no_compile),
    )
    try:
        decoder.decode_many(indices)
    finally:
        decoder.close()


if __name__ == "__main__":
    main()
