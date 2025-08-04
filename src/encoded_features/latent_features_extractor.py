import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import repeat, rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from nsd_access import NSDAccess
from PIL import Image
import json
import clip
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
    def __init__(self, img_idx, gpu, seed, captions_num, resolution, output_dir, nsd_dir):
        self.img_idx = img_idx
        self.gpu = gpu
        self.seed = seed
        self.captions_num = captions_num
        self.resolution = resolution
        self.output_dir = output_dir
        self.nsd_dir = nsd_dir
        self.device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
        self.nsda = NSDAccess(nsd_dir)

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # os.makedirs(os.path.join(output_dir, 'latent_features/init_latent'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f'latent_features/c_top{captions_num}_captions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'reconstructed_images_from_originals'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'original_stimulis'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'coco_captions'), exist_ok=True)

        seed_everything(seed)

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
    def __init__(self, img_idx, gpu, seed, captions_num, resolution, output_dir, nsd_dir, config_path, ckpt_path):
        super().__init__(img_idx, gpu, seed, captions_num, resolution, output_dir, nsd_dir)
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.batch_size = 1
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.strength = 0.65
        self.scale = 5.0

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

        for img_idx in tqdm(range(self.img_idx[0], self.img_idx[1])):
            img_str = f"{img_idx:06}"
            # if os.path.exists(os.path.join(self.output_dir, 'latent_features/init_latent', f"{img_str}.npy")):
                # logger.info(f"Skipping {img_str}, already processed.")
                # continue

            logger.info(f"Processing image {img_str}")
            img = self.nsda.read_images(img_idx)
            # Image.fromarray(img.astype(np.uint8)).save(os.path.join(self.output_dir, 'original_stimulis', f"{img_str}_original.png"))

            prompts = self.nsda.read_image_coco_info([img_idx], info_type='captions')
            captions = [p['caption'] for p in prompts]
            top_captions, _ = self.get_top_captions(img_idx, captions)

            self.save_json(os.path.join(self.output_dir, 'coco_captions', f'nsd_image_captions{self.captions_num}.json'), {img_str: captions})
            self.save_json(os.path.join(self.output_dir, 'coco_captions', f'nsd_best_captions{self.captions_num}.json'), {img_str: top_captions})

            # init_image = self.load_image_from_arr(img).to(self.device)
            # init_image = repeat(init_image, '1 ... -> b ...', b=self.batch_size)

            with torch.no_grad():
                with precision_scope("cuda"):
                    with self.model.ema_scope():
                        # init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))
                        # uc = self.model.get_learned_conditioning(self.batch_size * [""])
                        c = self.model.get_learned_conditioning(top_captions).mean(axis=0).unsqueeze(0)

                        # z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * self.batch_size).to(self.device))
                        # samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=self.scale, unconditional_conditioning=uc)
                        # x_samples = self.model.decode_first_stage(samples)
                        # x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
                        # x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
                        # Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(self.output_dir, 'reconstructed_images_from_originals', f"{img_str}_recon.png"))

            # np.save(os.path.join(self.output_dir, 'latent_features','init_latent', f"{img_str}.npy"), init_latent.cpu().numpy())
            np.save(os.path.join(self.output_dir, 'latent_features', f'c_top{self.captions_num}_captions', f"{img_str}.npy"), c.cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description="Extract features using Stable Diffusion.")
    parser.add_argument("--img_idx", nargs=2, type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--captions_num", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--output_dir", type=str, default="/mnt/outputs/")
    parser.add_argument("--nsd_dir", type=str, default="/mnt/datasets/nsd/")
    parser.add_argument("--config_path", type=str, default="diffusion_sd1/stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    parser.add_argument("--ckpt_path", type=str, default="models/stable-diffusion/sd-v1-4.ckpt")

    args = parser.parse_args()
    extractor = StableDiffusionExtractor(
        img_idx=args.img_idx,
        gpu=args.gpu,
        seed=args.seed,
        captions_num=args.captions_num,
        resolution=args.resolution,
        output_dir=args.output_dir,
        nsd_dir=args.nsd_dir,
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
    )
    extractor.extract_features()

if __name__ == "__main__":
    main()