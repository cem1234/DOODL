from __future__ import annotations

from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup


def _read_readme() -> str:
    readme_path = Path(__file__).with_name("README.md")
    try:
        return readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


setup(
    name="doodl",
    version="0.0.0",
    description="Official Implementation of DOODL (End-to-End Diffusion Latent Optimization Improves Classifier Guidance)",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    url="https://github.com/cem1234/DOODL",
    py_modules=[
        "doodl",
        "helper_functions",
    ],
    packages=(
        find_packages()
        + find_namespace_packages(include=["fgvc_ws_dan_helpers*"])
    ),
    include_package_data=True,
    python_requires=">=3.8",
    extras_require={
        "deps": [
            "numpy",
            "Pillow",
            "imageio",
            "tqdm",
            "omegaconf",
            "transformers",
            "open-clip-torch",
            "torch",
            "torchvision",
        ],
    },
)
