# Medical Imaging Model Robustness under Compiler Optimizations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Project Overview

This research-oriented project investigates a critical question at the intersection of AI systems and clinical deployment:

> **How do compiler-level optimizations affect the out-of-distribution (OOD) robustness of 3D medical imaging models?**

As we push for real-time performance in clinical settings (e.g., surgical navigation), we must ensure that optimization does not come at the cost of reliability on unpredictable, real-world data.

##  Key Features & Approach

- ** Model**: 3D U-Net for volumetric medical image segmentation (e.g., kidney & tumor segmentation on KiTS19).
- ** Optimization**: Leverages TVM for model compilation and acceleration.
- ** Research Focus**: Systematic evaluation of model performance on:
  - **Clean Test Data** (i.i.d. scenario)
  - **Perturbed Data** (OOD scenario), simulating real-world challenges:
    - Noise Injection
    - Motion Artifacts
    - Contrast Variations
- ** Analysis**: In-depth comparison of accuracy/robustness trade-offs across different optimization levels.

## Project Structure

```plaintext
Medical-OOD-Robustness/
├── data/               # Data handling scripts and utilities
├── models/             # 3D U-Net implementation and model definitions
├── optimization/       # TVM compilation and optimization scripts
├── evaluation/         # Robustness evaluation and perturbation pipelines
├── docs/               # Technical report and detailed findings
├── tests/              # Unit tests for critical components
├── requirements.txt    # Python dependencies
└── README.md          # This file

```

## Installation
1. clone the repository
   ```bash
   git clone https://github.com/your-username/Medical-OOD-Robustness.git
   cd Medical-OOD-Robustness
   ```
2. Set up enviroment
   ```bash
   conda create -n med-robust python=3.10
   conda activate med-robust
   pip install -r requirements.txt
   ```
3. Download Data
   - Obtain KiTS19 data from the [official website](https://kits19.grand-challenge.org/)
   - Place in data/kits19/
     
4. Run baseline training
   ```bash
   python train.py --config configs/baseline.yaml
   ```
## Preliminary Results

## Contributing
This is a research project in active development. Discussions, issues, and suggestions are welcome!

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
