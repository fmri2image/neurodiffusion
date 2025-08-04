# modal_app_pipeline.py

import modal

app = modal.App("brain2image-pipeline")

# ⬇️ Build your environment from requirements.txt
image = modal.Image.debian_slim().apt_install("git").pip_install_from_requirements("requirements.txt").pip_install("git+https://github.com/CompVis/taming-transformers.git@master")

# ⬇️ Mount your Modal volume (update volume name)
volume = modal.Volume.from_name("brainfmri2image")

# ⬇️ Common options used in all pipeline steps
COMMON_OPTS = {
    "image": image,
    "volumes": {"./": volume},
    # "gpu": "any",  # Optional, for GPU-requiring steps like reconstruction
    "timeout": 6000 # In seconds (if needed for long jobs)
}


# 🔧 STEP 1: Run NSD preprocessing
@app.function(**COMMON_OPTS)
def run_preprocessor(subject: str = "subj01", nsd_root:str = "datasets/nsd/", output_dir:str = "outputs/fmrifeatures", atlas:str = "streams"):
    from src.data.nsd_data_preprocessor import NSDPreprocessor

    processor = NSDPreprocessor(
        nsd_root=nsd_root,
        output_dir=output_dir,
        atlas_name=atlas
    )
    processor.run(subject, overwrite=True)

    print(f"✅ Finished preprocessing for {subject}")


# 🔧 STEP 2: Encode images to latents (placeholder)
@app.function(**COMMON_OPTS)
def encode_features(subject: str):
    print(f"🔐 Encoding features for {subject}...")
    # Import and call your encoder here (e.g., sd_encoder)
    # from src.encoding.sd_encoder import encode_subject
    # encode_subject(...)
    print(f"✅ Done encoding features for {subject}")


# 🔧 STEP 3: Train decoder (placeholder)
@app.function(**COMMON_OPTS)
def train_decoder(subject: str):
    print(f"🎯 Training model for {subject}...")
    # from src.decoding.my_decoder import train_model
    # train_model(...)
    print(f"✅ Finished training for {subject}")


# 🔧 STEP 4: Reconstruct & Evaluate (placeholder)
@app.function(**COMMON_OPTS)
def reconstruct_and_evaluate(subject: str):
    print(f"🧠 Reconstructing images for {subject}...")
    # import and run your reconstruction & evaluation here
    print(f"📊 Evaluation complete for {subject}")

