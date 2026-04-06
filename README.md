# Face Defense

Multi-layer face anti-spoofing and deepfake detection research framework using open-source models.

## Getting Started

### Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA support (recommended)
- Conda (Miniconda or Anaconda)

### Installation

```bash
# Create conda environment
conda create -n face-defense python=3.10 -y
conda activate face-defense

# Install dlib via conda (pip build fails on Windows due to cp949 encoding issue)
conda install -c conda-forge dlib -y

# Install remaining dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

