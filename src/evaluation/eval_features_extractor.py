import os
import glob
import argparse
from typing import List, Dict, Set, Optional, Iterable

import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

FEATURE_SUFFIXES = {
    "inception": ["_inception.npy"],
    "alexnet": ["_alexnet5.npy", "_alexnet12.npy", "_alexnet18.npy"],
    "clip": ["_clip.npy", "_clip_h6.npy", "_clip_h12.npy"],
}


class DecodedEvalFeatureExtractor:
    def __init__(
            self,
            subject: str,
            method: str,
            gpu: int = 0,
            output_dir: str = "/mnt/outputs",
            skip_existing: bool = True,
            features: Iterable[str] = ("inception", "alexnet", "clip"),  # was: List[str] | Set[str]
    ):
        """
        Args:
            subject: subj01 / subj02 / subj05 / subj07 ...
            method:  cvpr | text | gan | depth  (matches your folder naming)
            gpu:     CUDA device index
            decoded_glob: optional glob override for decoded image paths.
                          Default: f"{output_dir}/decoded/image-{method}/{subject}/samples/*"
            output_dir: base output dir; features saved to {output_dir}/identification/{method}/{subject}
            skip_existing: skip images if all requested outputs already exist for that image
            features: which feature families to compute: any of {"inception","alexnet","clip"}
        """
        self.subject = subject
        self.method = method
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)

        self.output_dir = output_dir

        # discover decoded images
        decoded_glob = os.path.join(
                self.output_dir, f"reconstructed_images_from_predictions", self.method, self.subject, "*")
        self.imglist: List[str] = sorted(glob.glob(decoded_glob))

        # output dir
        self.extracted_features_dir = os.path.join(self.output_dir, f"eval_extracted_features/{self.method}/{self.subject}/")
        os.makedirs(self.extracted_features_dir, exist_ok=True)

        # config
        self.skip_existing = skip_existing
        self.features: Set[str] = {f.lower() for f in features}

        # models / preprocessors (lazy-loaded)
        self._inception = None
        self._alexnet = None
        self._clip_model = None
        self._clip_processor = None

        # preprocessing for torchvision models
        self._preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ----------------------- model loaders -----------------------

    def _load_inception(self):
        if self._inception is not None:
            return self._inception
        model = torchvision.models.inception_v3(pretrained=True)
        model.eval().to(self.device)
        model = create_feature_extractor(model, return_nodes={'flatten': 'flatten'})
        self._inception = model
        return model

    def _load_alexnet(self):
        if self._alexnet is not None:
            return self._alexnet
        model = torchvision.models.alexnet(pretrained=True)
        model.eval().to(self.device)
        return_nodes = {
            'features.5': 'features.5',
            'features.12': 'features.12',
            'classifier.5': 'classifier.5',
        }
        model = create_feature_extractor(model, return_nodes=return_nodes)
        self._alexnet = model
        return model

    def _load_clip(self):
        if self._clip_model is not None and self._clip_processor is not None:
            return self._clip_model, self._clip_processor
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.to(self.device).eval()
        self._clip_model = model
        self._clip_processor = processor
        return model, processor

    # ----------------------- helpers -----------------------

    def _requested_suffixes(self) -> List[str]:
        suf = []
        for fam in self.features:
            suf.extend(FEATURE_SUFFIXES[fam])
        return suf

    def _all_outputs_exist(self, base_path_no_ext: str) -> bool:
        """Check whether all requested npy outputs exist for a given base path."""
        return all(os.path.exists(base_path_no_ext + s) for s in self._requested_suffixes())

    # ----------------------- extraction -----------------------

    def extract_one(self, img_path: str) -> Dict[str, np.ndarray]:
        """
        Extract requested features for a single image path.
        Returns a dict mapping name -> numpy array.
        """
        with Image.open(img_path) as im:
            image = im.convert("RGB")

        it = self._preprocess(image).unsqueeze(0).to(self.device)
        results: Dict[str, np.ndarray] = {}

        # ---- Inception ----
        if "inception" in self.features:
            inc = self._load_inception()
            with torch.inference_mode():
                out_inc = inc(it)['flatten']  # [1, 2048]
            results["inception"] = out_inc.detach().cpu().numpy().copy()

        # ---- AlexNet ----
        if "alexnet" in self.features:
            alex = self._load_alexnet()
            with torch.inference_mode():
                out_alex = alex(it)
            results["alexnet5"] = out_alex['features.5'].flatten().detach().cpu().numpy().copy()
            results["alexnet12"] = out_alex['features.12'].flatten().detach().cpu().numpy().copy()
            results["alexnet18"] = out_alex['classifier.5'].flatten().detach().cpu().numpy().copy()

        # ---- CLIP (ViT-L/14) ----
        # ---- CLIP (ViT-L/14) ----
        if "clip" in self.features:
            clip_model, clip_processor = self._load_clip()

            # Build pixel_values with the CLIP processor (handles resize/normalize/crop)
            inputs = clip_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.inference_mode():
                # Get vision hidden states (no text needed)
                vision_out = clip_model.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True
                )
                # Project pooled vision features to CLIP embedding space
                pooled = vision_out.pooler_output  # (B, hidden_size)
                image_embeds = clip_model.visual_projection(pooled)  # (B, projection_dim)

                # Save global image embedding
                results["clip"] = image_embeds.detach().cpu().numpy().copy()

                # Save selected hidden states (match your original indices)
                h6 = vision_out.hidden_states[6]
                h12 = vision_out.hidden_states[12]
                results["clip_h6"] = h6.flatten().detach().cpu().numpy().copy()
                results["clip_h12"] = h12.flatten().detach().cpu().numpy().copy()

        return results

    def run(self):
        print(f"[INFO] Now processing start for: {self.method} | features={sorted(self.features)}")
        if not self.imglist:
            print(f"[WARN] No images found for pattern.")
            return

        for img in tqdm(self.imglist, desc="img2feat", unit="img"):
            imgname = os.path.splitext(os.path.basename(img))[0]
            out_base = os.path.join(self.extracted_features_dir, imgname)

            if self.skip_existing and self._all_outputs_exist(out_base):
                continue  # already done

            feats = self.extract_one(img)

            # Save only what was computed
            if "inception" in feats:
                np.save(f"{out_base}_inception.npy", feats["inception"])
            if "alexnet5" in feats:
                np.save(f"{out_base}_alexnet5.npy", feats["alexnet5"])
                np.save(f"{out_base}_alexnet12.npy", feats["alexnet12"])
                np.save(f"{out_base}_alexnet18.npy", feats["alexnet18"])
            if "clip" in feats:
                np.save(f"{out_base}_clip.npy", feats["clip"])
                np.save(f"{out_base}_clip_h6.npy", feats["clip_h6"])
                np.save(f"{out_base}_clip_h12.npy", feats["clip_h12"])


def main():
    parser = argparse.ArgumentParser(description="Extract features for decoded images (Inception, AlexNet, CLIP).")
    parser.add_argument("--gpu", required=True, type=int, help="CUDA device index")
    parser.add_argument("--subject", required=True, type=str,
                        help="subj01 | subj02 | subj05 | subj07 ...")
    parser.add_argument("--method", required=True, type=str,
                        help="cvpr | text | gan | depth")
    parser.add_argument("--output-dir", type=str, default="/mnt/outputs",
                        help="Base output dir (features to {output}/identification/{method}/{subject})")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Do not skip images with all requested outputs already saved.")
    parser.add_argument("--features", nargs="+", choices=["inception", "alexnet", "clip"],
                        default=["inception", "alexnet", "clip"],
                        help="Which feature families to compute.")
    args = parser.parse_args()

    extractor = DecodedEvalFeatureExtractor(
        subject=args.subject,
        method=args.method,
        gpu=args.gpu,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip_existing,
        features=args.features,
    )
    extractor.run()


if __name__ == "__main__":
    main()
