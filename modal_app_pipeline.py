# modal_stub_pipeline.py

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
    "gpu": "T4",  # Optional, for GPU-requiring steps like reconstruction
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
        packages_path: str = "/mnt/packages/"
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
        packages_path=packages_path
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


# 🔧 STEP 4: Train Model (placeholder)
@app.function(**COMMON_OPTS)
def train_model(subject: str, roi: str, target: str, model: str = "all", pca_dim: int = 512):
    import sys
    sys.path.append("/mnt/")
    roi = roi.split(",")

    from src.models.regression import RidgeRegression, KernelRidgeRegression, MLPRegression

    models = {
        'ridge': RidgeRegression,
        'kernelridge': lambda **kwargs: KernelRidgeRegression(**kwargs, pca_dim=pca_dim),
        'mlp': lambda **kwargs: MLPRegression(**kwargs, pca_dim=pca_dim)
    }

    if model == 'all':
        selected_models = models.values()
    else:
        selected_models = [models[model]]

    for model_class in selected_models:
        reg_model = model_class(
            target=target,
            roi=roi,
            subject=subject,
            fmri_dir="/mnt/outputs/fmri_features/",
            feat_dir="/mnt/outputs/latent_features",
            models_dir="/mnt/outputs/models"
        )
        model_path, scores, mean_rs = reg_model.train()
        print(
            f"{model_class.__name__ if hasattr(model_class, '__name__') else 'MLPRegression'} saved to: {model_path}, Mean correlation: {mean_rs:.3f}")


@app.function(image=image, gpu="T4", timeout=60 * 30, volumes={"/mnt": volume})
def run_decoder(imgidx: str, method: str, subject: str):
    import sys
    sys.path.append("/mnt")
    from src.decoding.diffusion_decoding import DiffusionDecoder

    decoder = DiffusionDecoder(
        imgidx=imgidx,
        method=method,
        subject=subject,
        gpu=0,
        seed=42,
        nsd_dir="/mnt/datasets/nsd/",
        output_dir="/mnt/outputs",
        config_path="/mnt/packages/stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
        ckpt_path="/mnt/packages/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"
    )
    decoder.decode()


# 🔧 STEP 4: Reconstruct & Evaluate (placeholder)
@app.function(**COMMON_OPTS)
def reconstruct_and_evaluate(subject: str):
    print(f"🧠 Reconstructing images for {subject}...")
    # import and run your reconstruction & evaluation here
    print(f"📊 Evaluation complete for {subject}")
