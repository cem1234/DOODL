# DOODL-Main Colab notebook overview

This note summarizes the major cells in the `DOODL-Main` Colab notebook so it is easier to navigate when setting up or running experiments.

## Runtime and environment prep
- Start the notebook on a GPU runtime and mount Google Drive under `/content/drive` (Cell 0).
- Authenticate to Hugging Face using a token stored in `/content/drive/MyDrive/DOODL-experiments/HFToken.txt`; the notebook reads the token and calls `huggingface_hub.login`, printing a warning if the file is missing (Cell 2).
- Install core Python dependencies (Transformers, Tokenizers, OpenCLIP, LPIPS, etc.) and clone the `mechint` branch of `cem1234/DOODL` into `/content/DOODL`; the token is also written to `hf_auth` for reuse (Cells 3–4).
- Import the local `doodl.py` to confirm the forked code (Cell 5).

## Experiment bookkeeping
- Define `source_class`, `target_class`, timestamped `run_name`, and create a Drive-backed `run_dir` under `/content/drive/MyDrive/DOODL-experiments/Runs` for logs and images (Cell 6).
- Verify library versions, compiled torchvision ops, and that the vendored `my_half_diffusers` package is importable from the cloned repo (Cell 8).
- Point to your MechInt probe assets on Drive: `MI_CODE_DIR` is appended to `sys.path`, `PROBE_ROOT` should contain probe results, and `REF_IMAGE` should point to the reference image used for LPIPS or inversion (Cells 9–10).

## BCos environment and helpers
- Install the `bcos` package and import utility helpers such as `ProbeResultsRepository`, `build_subspace_from_probe`, and `project_activations` (Cells 13–14).
- `load_env(model_name="bcos")` loads either a GELU or BCos ViT-C Tiny backbone, builds `MaskedModelModal` over encoder layers, moves everything to CUDA if available, and returns `(device, model, masked_model_modal, idx2label)` for downstream guidance (Cell 15).
- `preprocess_bcos_tensor` normalizes tensors to [0,1], resizes to 256px, center-crops to 224px, concatenates the inverted channels, and yields a batch-ready tensor for BCos-based probe scoring (Cell 16).

## Wiring probes and reference images
- Call `load_env`, locate the latest probe for `target_class` via `ProbeResultsRepository.sorted_probes`, rebuild the probe subspace, and prepare a normalized `unit_probe` (Cell 18).
- Load the reference image, convert it to tensors in [0,1] and [-1,1], and initialize an LPIPS model with a VGG backbone for anti-drift losses (Cell 19).

## MechInt-guided editing (with a source image)
- Build `model_guidance_dict` with BCos wrappers, projection utilities, probe vectors (both normalized and raw), optional reference tensor placeholders, LPIPS model, token aggregation strategy, component weights (cross-entropy, probe, LPIPS, and latent L2), plus latent prior parameters (Cell 21).
- Extend the guidance dict with `ref_image_path` when you want LPIPS against the stored reference image (Cell 23).
- Configure `exp_cfg` for `doodl.gen`, pointing `save_dir` to Drive, tuning traversal hyperparameters (e.g., `num_traversal_steps=60`, `steps=50`, `mix_weight=0.93`), disabling latent renormalization, and setting `source_im` to trigger inversion around the reference (Cell 23).
- Persist a JSON manifest that captures serializable parts of the experiment config and the weight settings, then run `doodl.gen` to write images and CSV logs into `run_dir` (Cell 23).

## MechInt-guided novel generation (random latents)
- Start from the base guidance dictionary but drop LPIPS/z_l2 pressures, adjust weights (e.g., enable probe weight), and optionally set latent priors to limit drift (Cell 26).
- Configure `exp_cfg_random` to re-enable latent renormalization, omit `source_im`, and optionally seed the latent initialization and randomness for reproducibility (Cell 27).
- A separate example shows another novel-generation run with tuned hyperparameters (e.g., `num_traversal_steps=80`, `grad_scale=0.02`, `clip_grad_val=3e-4`) and `optimize_first_edict_image_only=True` for stability (Cell 30).

## Reading and evaluating outputs
- Later cells load generated image directories from Drive and offer a quick CSV/Matplotlib visualization of per-step metrics such as cross-entropy, probe alignment, LPIPS, and prior losses (Cells 31–33).
