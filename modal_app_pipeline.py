# modal_app_pipeline.py
from typing import Union, List, Iterable
import re

import modal

app = modal.App("brain2image-pipeline")

# ⬇️ Build your environment from requirements.txt
image = modal.Image.debian_slim().apt_install("git"). \
    pip_install_from_requirements("requirements.txt") \
    .env({"PYTHONPATH": "/mnt/packages"}) \
    .pip_install("numpy==1.24.4").pip_install("nibabel==4.0.2").pip_install("git+https://github.com/openai/CLIP.git")
# ⬇️ Mount your Modal volume (update volume name)
volume = modal.Volume.from_name("brainfmri2image")

# ⬇️ Common options used in all pipeline steps
COMMON_OPTS = {
    "image": image,
    "volumes": {"/mnt/": volume},
    # "gpu": "T4",  # Optional, for GPU-requiring steps like reconstruction
    "timeout": 50000  # In seconds (if needed for long jobs)
}


# 🔧 STEP 1: Run NSD preprocessing
@app.function(**COMMON_OPTS)
def run_preprocessor(subject: str = "subj02", output_dir: str = "/mnt/outputs/fmri_features/",
                     nsd_root: str = "/mnt/datasets/nsd/", atlas: str = "streams"):
    import sys
    sys.path.append("/mnt/")  # Now /mnt/src is visible to Python

    from src.data_preprocessing.nsd_fmri_preprocessor import NSDPreprocessor

    processor = NSDPreprocessor(
        output_dir=output_dir,
        atlas_name=atlas,
        nsd_root=nsd_root
    )
    processor.run(subject)

    print(f"✅ Finished preprocessing for {subject}")


# 🔧 STEP 2: Encode image features (Stable Diffusion)
@app.function(**COMMON_OPTS, gpu="T4")
def encode_features(
        gpu: int = 0,
        captions_num: int = 3,
        output_dir: str = "/mnt/outputs/",
        subject: str = "subj02",
        nsd_root: str = "/mnt/datasets/nsd/",
        packages_path: str = "/mnt/packages/",
        recon_img_flag=False
):
    import sys
    sys.path.append("/mnt/")  # Now /mnt/src is visible to Python

    from src.data_preprocessing.latent_features_extractor import StableDiffusionExtractor

    extractor = StableDiffusionExtractor(
        img_idx=None,
        gpu=gpu,
        captions_num=captions_num,
        output_dir=output_dir,
        subject=subject,
        nsd_root=nsd_root,
        packages_path=packages_path,
        recon_img_flag=recon_img_flag
    )
    extractor.extract_features()


# 🔧 STEP 3: Split latent features into train/test sets
@app.function(**COMMON_OPTS)
def split_latent_features(subject: str = "subj02", featname: str = "init_latent", use_stim: str = "each",
                          nsd_dir: str = "/mnt/datasets/nsd", output_dir: str = "/mnt/outputs"):
    import sys
    sys.path.append("/mnt/")  # ensure /mnt/src and packages are visible

    from src.data_preprocessing.latent_features_splitter import LatentsSplitter  # adjust path to match your structure

    splitter = LatentsSplitter(
        featname=featname,
        use_stim=use_stim,
        subject=subject,
        nsd_dir=nsd_dir,
        output_dir=output_dir
    )

    splitter.split_latent_features()

    print(f"✅ Latent feature split complete for {subject} ({featname}, {use_stim})")


# 🔧 STEP 4 : Train Model Enhanced
@app.function(**COMMON_OPTS)
def train_model(
        subject: str,
        roi,  # list or comma/space-separated str
        target: str,
        model: str = "all",  # 'ridge' | 'kernelridge' | 'mlp' | 'all'
        pca_dim: int = 512,
        use_pca: bool = False,
        betas_flag: str = "each",  # 'each' or 'ave'
        ridge_kernel: str = "rbf",
        fmri_dir: str = "/mnt/outputs/fmri_features/",
        feat_dir: str = "/mnt/outputs/latent_features/",
        models_dir: str = "/mnt/outputs/saved_models/",
):
    import sys
    from functools import partial
    sys.path.append("/mnt/")

    from src.model_training.regression_model_trainer import (
        RidgeRegression,
        KernelRidgeRegression,
        MLPRegression,
    )

    # normalize ROI to list
    if isinstance(roi, str):
        roi_list = [r for r in (roi.split(",") if "," in roi else roi.split()) if r]
    else:
        roi_list = list(roi)

    # coerce use_pca if it arrives as string
    if isinstance(use_pca, str):
        use_pca = use_pca.strip().lower() in {"1", "true", "yes", "y", "on"}

    models = {
        "ridge": partial(RidgeRegression, pca_dim=pca_dim),
        "kernelridge": partial(KernelRidgeRegression, ridge_kernel=ridge_kernel, pca_dim=pca_dim),
        "mlp": partial(MLPRegression, pca_dim=pca_dim),
    }

    if model != "all" and model not in models:
        raise ValueError(f"Unknown model '{model}'. Choose one of {list(models.keys()) + ['all']}.")

    # keep names for logging/results
    selected_items = list(models.items()) if model == "all" else [(model, models[model])]

    results = []
    for name, ctor in selected_items:
        try:
            print(f"▶ Training {name} | subj={subject} | roi={roi_list} | target={target} "
                  f"| use_pca={use_pca} | pca_dim={pca_dim} | betas_flag={betas_flag} | kernel={ridge_kernel}")
            reg_model = ctor(
                target=target,
                roi=roi_list,
                subject=subject,
                fmri_dir=fmri_dir,
                feat_dir=feat_dir,
                models_dir=models_dir,
                betas_flag=betas_flag,
                use_pca=use_pca,
            )
            model_path, scores, mean_rs = reg_model.train()
            print(f"✔ {reg_model.__class__.__name__} saved: {model_path} | mean corr={mean_rs:.3f}")
            results.append({
                "model": name,
                "path": model_path,
                "mean_corr": float(mean_rs),
                "roi": roi_list,
                "target": target,
                "use_pca": use_pca,
                "pca_dim": pca_dim,
                "kernel": ridge_kernel if name == "kernelridge" else None,
            })
        except Exception as e:
            print(f"✖ Failed {name}: {e}")
            results.append({"model": name, "error": str(e)})

    return results


# STEP 5: Run decoder.
@app.function(**COMMON_OPTS)
def run_decoder(
        imgidx: str = "0",  # accept "7" or "1 10" (inclusive range)
        method: str = "kernelridgeregression",
        subject: str = "subj05",
        roi_init_latent: str = "early ventral",  # space or comma-separated OK
        roi_c: str = "early ventral",
        captions_type: str = "c_top1_captions",
        kernel_used_init_latent: str = "rbf",
        kernel_used_c: str = "poly",
        # IMPORTANT: separate PCA tags (strings like "pca512" or "nopca")
        pca_dim_c: str = "pca512",
        pca_dim_init_latent: str = "pca512",
        # IMPORTANT: pass both betas flags
        betas_flag_c: str = "each",
        betas_flag_init_latent: str = "each",
        # decoding params
        seed: int = 42,
        ddim_steps: int = 50,
        strength: float = 0.8,
        scale: float = 5.0,
        n_iter: int = 5,
        sample_bs: int = 1,  # batch multiple variations per image
        # paths / device
        gpu: int = 0,
        nsd_dir: str = "/mnt/datasets/nsd",
        output_dir: str = "/mnt/outputs",
        packages_dir: str = "/mnt/packages",
        # speedup toggles
        no_xformers: bool = False,  # set True to disable xFormers attempt
        no_compile: bool = False,  # set True to disable torch.compile attempt
):
    import sys
    sys.path.append("/mnt")
    from src.decoding.diffusion_decoding import DiffusionDecoder

    # normalize ROI strings -> list
    def _normalize_roi(val):
        if isinstance(val, (list, tuple)):
            return list(val)
        if "," in val:
            return [v.strip() for v in val.split(",") if v.strip()]
        return [v for v in val.split() if v]

    # parse imgidx string -> list of indices
    def _parse_imgidx(val: str):
        toks = [int(x) for x in val.replace(",", " ").split()]
        if len(toks) == 1:
            return [toks[0]]
        if len(toks) == 2:
            s, e = toks
            if s > e:
                s, e = e, s
            return list(range(s, e + 1))  # inclusive
        raise ValueError("imgidx: pass one number or two numbers for an inclusive range (e.g., '7' or '1 10').")

    indices = _parse_imgidx(imgidx)

    decoder = DiffusionDecoder(
        method=method,
        subject=subject,
        gpu=gpu,  # Modal gives you one GPU; inside container it's index 0
        nsd_dir=nsd_dir,
        output_dir=output_dir,
        packages_path=packages_dir,  # matches constructor name
        roi_init_latent=_normalize_roi(roi_init_latent),
        roi_c=_normalize_roi(roi_c),
        captions_type=captions_type,
        kernel_used_c=kernel_used_c,
        kernel_used_init_latent=kernel_used_init_latent,
        pca_dim_c=pca_dim_c,
        pca_dim_init_latent=pca_dim_init_latent,
        betas_flag_c=betas_flag_c,
        betas_flag_init_latent=betas_flag_init_latent,
        seed=seed,
        ddim_steps=ddim_steps,
        strength=strength,
        scale=scale,
        n_iter=n_iter,
        sample_bs=sample_bs,
        try_xformers=(not no_xformers),
        try_compile=(not no_compile),
    )
    try:
        decoder.decode_many(indices)
    finally:
        decoder.close()


# 🔧 STEP 6: Extract features from decoded images (Inception, AlexNet, CLIP)
@app.function(**COMMON_OPTS)
def run_decoded_eval_features(
        subject: str = "subj02",
        method: str = "kernelridgeregression_init_latent_info -> (roi_early_pca_nopca_kernel_rbf)__c_info -> ("
                      "roi_ventral_pca_pca1024_kernel_poly_captions_c_top1_captions)",
        gpu: int = 0,
        output_dir: str = "/mnt/outputs",
        no_skip_existing: bool = False,  # set True to force recompute
        features: str = "alexnet,clip",
):
    import sys
    sys.path.append("/mnt")
    from src.evaluation.eval_features_extractor import DecodedEvalFeatureExtractor
    feats = [f.strip().lower() for f in re.split(r"[,\s]+", features) if f.strip()]
    allowed = {"inception", "alexnet", "clip"}
    unknown = [f for f in feats if f not in allowed]
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown}. Allowed: {sorted(allowed)}")

    extractor = DecodedEvalFeatureExtractor(
        subject=subject,
        method=method,
        gpu=gpu,
        output_dir=output_dir,
        skip_existing=(not no_skip_existing),
        features=feats
    )
    extractor.run()


# 🔧 STEP 7: Calculate the identification accuracy metric
@app.function(**COMMON_OPTS)
def run_evaluate_identification_acc(
    subject: str = "subj02",
    method: str = "kernelridgeregression_init_latent_info -> (roi_early_pca_nopca_kernel_rbf)__c_info -> ("
                      "roi_ventral_pca_pca1024_kernel_poly_captions_c_top1_captions)",
    features: str = "alexnet12 alexnet18 clip clip_h6 clip_h12",            # e.g. "all" or "inception clip clip_h12"
    output_dir: str = "/mnt/outputs",
    n_images: int = 982,
    n_rep: int = None,         # leave None to auto-detect
    img_range: str = "",              # e.g. "0 981" or "100,199"
    non_strict: bool = False,         # True => skip missing files
    chunk_size: int = 64,             # batching for comparisons
):
    """
    Evaluate identification accuracy for one or many features.

    Examples:
      # all features
      modal run modal_app_pipeline.py::run_evaluate_identification_acc \
        --subject subj02 --method kernelridgeregression --features all --img-range "0 981"

      # subset of features
      modal run modal_app_pipeline.py::run_evaluate_identification_acc \
        --subject subj02 --method kernelridgeregression --features "inception clip clip_h12" --img-range "0 981"
    """
    import sys
    sys.path.append("/mnt")

    # Import your evaluator class (module path should match where you saved it)
    from src.evaluation.identification_accuracy import IdentificationAccuracyEvaluator

    ALLOWED = ["inception", "alexnet5", "alexnet12", "alexnet18", "clip", "clip_h6", "clip_h12"]

    def _parse_features(s: str):
        toks = [t.strip() for t in s.replace(",", " ").split() if t.strip()]
        if not toks or toks == ["all"]:
            return ALLOWED
        bad = [t for t in toks if t not in ALLOWED]
        if bad:
            raise ValueError(f"Unknown features {bad}; allowed: {ALLOWED} (or 'all').")
        return toks

    def _parse_range(s: str):
        if not s:
            return None
        toks = [t for t in s.replace(",", " ").split() if t]
        if len(toks) != 2:
            raise ValueError("img_range must be two ints: 'START END' or 'START,END'")
        a, b = int(toks[0]), int(toks[1])
        return (min(a, b), max(a, b))

    feats = _parse_features(features)
    rng = _parse_range(img_range)

    results: dict[str, float] = {}
    for feat in feats:
        evaluator = IdentificationAccuracyEvaluator(
            subject=subject,
            method=method,
            feature=feat,
            output_dir=output_dir,
            n_images=n_images,
            n_rep=n_rep,
            img_range=rng,
            strict=not non_strict,
            chunk_size=chunk_size,
        )
        acc = evaluator.run()
        results[feat] = acc

    print("\n=== Identification accuracy summary ===")
    for f in ALLOWED:
        if f in results:
            print(f"{f:10s}: {results[f]:.3f}")

    return {"subject": subject, "method": method, "accuracies": results}
