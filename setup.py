from setuptools import find_packages, setup

setup(
    name="pansr",
    version="1.0.0",
    description="PanSR: Panoptic Segmentation of maritime scenes (LaRS).",
    packages=find_packages(include=["pansr", "pansr.*"]),
    python_requires=">=3.8",
    # torch, torchvision and detectron2 are installed separately (see setup.sh / README),
    # because they require platform/CUDA-specific wheels or a from-source build.
)
