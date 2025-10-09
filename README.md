# 3D Medical Imaging Visualization Platform

A comprehensive visualization tool for exploring 3D medical imaging data and AI segmentation results. This platform enables interactive exploration of kidney CT scans with model predictions and expert annotations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Project Overview

This project provides an intuitive interface for visualizing 3D medical imaging data, specifically designed for kidney CT scans from the KITS23 dataset. It allows researchers and medical professionals to explore imaging data, compare AI model predictions with ground truth annotations, and understand model behavior through interactive visualization.


##  Key Features & Approach

- **3D Volume Visualization**: Browse through CT scan slices with intuitive navigation
- **AI Model Integration**: View segmentation predictions from trained 3D U-Net models
- **Annotation Comparison**: Compare model outputs with expert radiologist annotations
- **Multi-case Support**: Explore multiple patient cases from the KITS23 dataset
- **Interactive Analysis**: Adjust visualization parameters in real-time


## Dataset

The project uses the KITS23 (Kidney Tumor Segmentation Challenge 2023) dataset, containing:
- 73 preprocessed CT volumes
- Expert annotations for kidneys, tumors, and cysts
- Standardized dimensions: 128×512×512 (depth×height×width)

## Model Architecture

The segmentation is performed by a custom 3D U-Net model:
- Input: 1×128×512×512 CT volumes
- Output: 4-class segmentation (background, kidney, tumor, cyst)
- Optimized for medical imaging with efficient memory usage

## Installation

```bash
git clone https://github.com/yourusername/medical-vision-explorer.git
cd medical-vision-explorer
conda create -n med-vis python=3.8
conda activate med-vis
pip install -r requirements.txt
```

## Usage
1. Basic Visualisation
   ```bash
   from medical_viewer import MedicalViewer
   # Load and display a case
   viewer = MedicalViewer()
   viewer.load_case('case_00123')
   viewer.show_slice(64)  # Display middle slice
   ```
2. Interactive Exploration
   ```bash
   # Launch interactive viewer
   viewer.launch_interactive()
   ```

## Project Structure

```plaintext
medical-vision-explorer/
├── medical_viewer.py          # Main visualization class
├── model_integration.py       # AI model loading and inference
├── data_loader.py            # Medical data loading utilities
├── preprocessed_data/        # Processed CT volumes and masks
├── trained_models/           # Pre-trained segmentation models
└── examples/                 # Usage examples and demos

```

## Requirements
```bash
   - Python 3.8+
   - Pytorch
   - Numpy
   - Matplotlib
   - Plotly (for interactive features)
   - MONAI (medical imaging utilities)
```

## Acknowledgments
- KITS23 Challenge organizers for the dataset
- MONAI team for medical imaging utilities
- Contributors to the 3D U-Net architecture

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
