from setuptools import setup, find_packages

setup(
    name="face-defense",
    version="0.2.0",
    packages=find_packages(include=["shared*", "antispoof*", "deepfake*", "emotion*"]),
    python_requires=">=3.9",
)
