import argparse
import os
from struct import pack
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from nsd_access import NSDAccess
from PIL import Image
import json
import clip
from einops import repeat, rearrange
import logging
from abc import ABC, abstractmethod

import sys;

sys.path.append("/mnt/packages/stable-diffusion")
sys.path.append("/mnt/packages/taming-transformers")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureExtractor")


class FeatureExtractor(ABC):
    def __init__(self, img_idx, gpu, captions_num, output_dir, subject, nsd_root, packages_path, recon_img_flag):
        self.subject = subject
        self.img_idx = img_idx
        self.gpu = gpu
        self.captions_num = captions_num
        self.output_dir = output_dir
        self.nsd_dir = nsd_root
        self.packages_path = packages_path
        self.device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
        self.nsda = NSDAccess(self.nsd_dir)
        self.recon_img_flag = recon_img_flag

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        os.makedirs(os.path.join(output_dir, f'latent_features/extracted_latents/init_latent/{self.subject}'),
                    exist_ok=True)
        os.makedirs(
            os.path.join(output_dir, f'latent_features/extracted_latents/c_top{self.captions_num}_captions/{self.subject}'),
            exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'reconstructed_images_from_originals/{self.subject}'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'original_images/{self.subject}'), exist_ok=True)

        seed_everything(42)

    def load_image_from_arr(self, img_arr):
        image = Image.fromarray(img_arr).convert("RGB")
        image = image.resize((self.resolution, self.resolution), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        return 2. * torch.from_numpy(image) - 1.

    def get_top_captions(self, img_idx, captions):
        image_np = np.squeeze(self.nsda.read_images(img_idx)).astype(np.uint8)
        image = self.clip_preprocess(Image.fromarray(image_np)).unsqueeze(0).to(self.device)
        text = clip.tokenize(captions).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            similarities = (image_features @ text_features.T).squeeze(0)
        top_indices = similarities.topk(k=self.captions_num).indices.tolist()
        return [captions[i] for i in top_indices], similarities[top_indices].tolist()

    def save_json(self, path, update_dict):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data.update(update_dict)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @abstractmethod
    def extract_features(self):
        pass


class StableDiffusionExtractor(FeatureExtractor):
    def __init__(self, img_idx, gpu, captions_num, output_dir, subject, nsd_root, packages_path, recon_img_flag):
        super().__init__(img_idx, gpu, captions_num, output_dir, subject, nsd_root, packages_path, recon_img_flag)

        self.config_path = os.path.join(self.packages_path,
                                        "stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
        self.ckpt_path = os.path.join(self.packages_path,
                                      "stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt")

        self.batch_size = 1
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.strength = 0.65
        self.scale = 5.0
        self.resolution = 320

        config = OmegaConf.load(self.config_path)
        self.model = self.load_model(config, self.ckpt_path)
        self.sampler = DDIMSampler(self.model)
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)
        assert 0. <= self.strength <= 1., 'can only work with strength in [0.0, 1.0]'

    def load_model(self, config, ckpt):
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        model.to(self.device).eval()
        return model

    def extract_features(self):
        precision_scope = autocast if torch.cuda.is_available() else nullcontext
        t_enc = int(self.strength * self.ddim_steps)

        if not self.img_idx:
            stim_file = os.path.join(
                self.output_dir,
                "fmri_features",
                self.subject,
                f"{self.subject}_stims_each.npy"
            )
            if not os.path.exists(stim_file):
                raise FileNotFoundError(f"No img_idx provided and file not found: {stim_file}")
            img_indices = np.load(stim_file).tolist()
            print(f"[INFO] Loaded {len(img_indices)} image indices from {stim_file}")
        else:
            img_indices = list(range(self.img_idx[0], self.img_idx[1]))
            print(f"[INFO] Using provided range: {self.img_idx[0]} → {self.img_idx[1] - 1}")

        for img_idx in tqdm(img_indices):
            img_str = f"{img_idx:06}"
            logger.info(f"Processing image {img_str}")

            z_path = os.path.join(self.output_dir,
                                  f'latent_features/extracted_latents/init_latent/{self.subject}/{img_str}.npy')
            c_path = os.path.join(self.output_dir,
                                  f'latent_features/extracted_latents/c_top{self.captions_num}_captions/{self.subject}/{img_str}.npy')
            orig_path = os.path.join(self.output_dir, f"original_images/{self.subject}/{img_str}_original.png")
            recon_path = os.path.join(self.output_dir,
                                      f'reconstructed_images_from_originals/{self.subject}/{img_str}.png')

            need_z = not os.path.exists(z_path)
            need_c = not os.path.exists(c_path)
            need_orig = self.recon_img_flag and (not os.path.exists(orig_path))
            need_recon = self.recon_img_flag and (not os.path.exists(recon_path))

            if not (need_z or need_c or need_orig or need_recon):
                logger.info(f"[{img_str}] All artifacts present → skip.")
                continue

            # Load NSD image once if needed for z or orig
            nsd_img = None
            if need_z or need_orig:
                nsd_img = self.nsda.read_images(img_idx)

            # --- c (conditioning) ---
            c = None
            if need_c:
                prompts = self.nsda.read_image_coco_info([img_idx], info_type='captions')
                captions = [p['caption'] for p in prompts]
                top_captions, _ = self.get_top_captions(img_idx, captions)
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with self.model.ema_scope():
                            c = self.model.get_learned_conditioning(top_captions).mean(dim=0, keepdim=True)
                np.save(c_path, c.detach().cpu().numpy())
            elif need_recon:
                c = torch.from_numpy(np.load(c_path)).to(self.device)

            # --- z (init_latent) ---
            init_latent = None
            if need_z:
                if nsd_img is None:
                    nsd_img = self.nsda.read_images(img_idx)
                init_image = self.load_image_from_arr(nsd_img).to(self.device)
                init_image = repeat(init_image, '1 ... -> b ...', b=self.batch_size)
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with self.model.ema_scope():
                            init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))
                np.save(z_path, init_latent.detach().cpu().numpy())
            elif need_recon:
                init_latent = torch.from_numpy(np.load(z_path)).to(self.device)

            # --- original image (optional) ---
            if need_orig:
                Image.fromarray(nsd_img.astype(np.uint8)).save(orig_path)

            # --- reconstruction (optional) ---
            if need_recon:
                # ensure both are loaded
                if c is None:
                    c = torch.from_numpy(np.load(c_path)).to(self.device)
                if init_latent is None:
                    init_latent = torch.from_numpy(np.load(z_path)).to(self.device)

                with torch.no_grad():
                    with precision_scope("cuda"):
                        with self.model.ema_scope():
                            uc = self.model.get_learned_conditioning(self.batch_size * [""])
                            z_enc = self.sampler.stochastic_encode(
                                init_latent, torch.tensor([t_enc] * self.batch_size).to(self.device)
                            )
                            samples = self.sampler.decode(
                                z_enc, c, t_enc,
                                unconditional_guidance_scale=self.scale,
                                unconditional_conditioning=uc
                            )
                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
                            x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(recon_path)

def main():
    parser = argparse.ArgumentParser(description="Extract features using Stable Diffusion.")
    parser.add_argument("--img_idx", nargs=2, type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--captions_num", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="/mnt/outputs/")
    parser.add_argument("--config_path", type=str,
                        default="diffusion_sd1/stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--ckpt_path", type=str, default="models/stable-diffusion/sd-v1-4.ckpt")
    parser.add_argument("--nsd_root", type=str, default="/mnt/datasets/nsd/")
    parser.add_argument("--packages_path", type=str, default="/mnt/packages/")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--images_save_flag", action="store_true",
                        help="Save original & reconstructed images when missing.")

    args = parser.parse_args()
    extractor = StableDiffusionExtractor(
        img_idx=args.img_idx,
        gpu=args.gpu,
        captions_num=args.captions_num,
        output_dir=args.output_dir,
        packages_path=args.packages_path,
        nsd_root=args.nsd_root,
        subject=args.subject,
        recon_img_flag=args.images_save_flag
    )
    extractor.extract_features()


if __name__ == "__main__":
    main()
