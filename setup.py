from setuptools import setup, find_packages

from python_coreml_stable_diffusion._version import __version__

setup(
    name='python_coreml_stable_diffusion',
    version=__version__,
    url='https://huggingface.co/Guernika/CoreMLStableDiffusion',
    description="Run Stable Diffusion on Apple Silicon with Guernika",
    author='Guernika',
    install_requires=[
        "coremltools>=6.1",
        "diffusers[torch]",
        "torch",
        "transformers",
        "scipy",
        "numpy<1.24",
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
