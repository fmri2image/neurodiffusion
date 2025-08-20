# modal_app_pipeline.py

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
@app.function(**COMMON_OPTS)
def encode_features(
        gpu: int = 0,
        captions_num: int = 1,
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
    model: str = "all",          # 'ridge' | 'kernelridge' | 'mlp' | 'all'
    pca_dim: int = 512,
    use_pca: bool = False,
    betas_flag: str = "each",    # 'each' or 'ave'
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
@app.function(**COMMON_OPTS)  # reuse COMMON_OPTS and add GPU here
def run_decoder(
        imgidx: int = 0,
        method: str = "kernelridgeregression",
        subject: str = "subj05",
        roi_init_latent: str = "early ventral",  # or space/comma-separated list
        roi_c: str = "early ventral",
        captions_type: str = "c_top1_captions",
        pca_dim: str = "pca512",
        seed: int = 42,
        ddim_steps: int = 50,
        strength: float = 0.8,
        scale: float = 5.0,
        n_iter: int = 5,
        kernel_used: str = "dummy",

):
    import sys
    sys.path.append("/mnt")
    from src.decoding.diffusion_decoding_enhanced import DiffusionDecoder

    # Allow "early ventral" or "early,ventral" inputs
    def _normalize_roi(val):
        if isinstance(val, (list, tuple)):
            return list(val)
        if "," in val:
            return [v.strip() for v in val.split(",") if v.strip()]
        return [v for v in val.split() if v]

    decoder = DiffusionDecoder(
        imgidx=imgidx,
        method=method,  # e.g., "kernelridge", "ridge", "mlp"
        subject=subject,
        gpu=0,  # Modal assigns one GPU; index 0 inside the container
        nsd_dir="/mnt/datasets/nsd",
        output_dir="/mnt/outputs",
        packages_path="/mnt/packages",
        roi_init_latent=_normalize_roi(roi_init_latent),
        roi_c=_normalize_roi(roi_c),
        captions_type=captions_type,
        kernel_used=kernel_used,
        pca_dim=pca_dim,
        seed=seed,
        ddim_steps=ddim_steps,
        strength=strength,
        scale=scale,
        n_iter=n_iter,
    )
    try:
        decoder.decode()
    finally:
        decoder.close()


# 🔧 STEP 6: Reconstruct & Evaluate (placeholder)
@app.function(**COMMON_OPTS)
def reconstruct_and_evaluate(subject: str):
    print(f"🧠 Reconstructing images for {subject}...")
    # import and run your reconstruction & evaluation here
    print(f"📊 Evaluation complete for {subject}")
