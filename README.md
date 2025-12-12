# Detecting and Defending Against Facial Image Cloaks in Recognition Pipelines

This project implements an end to end experimental pipeline to evaluate facial image cloaking (Fawkes) and lightweight defenses in a face recognition workflow.

The goal is to measure how much cloaking reduces recognition performance, and how much simple “purification” steps recover accuracy while keeping runtime practical.

# What This Repo Contains
- Dataset prep scripts (LFW face crops)
- Edited image generation (JPEG recompression + blur)
- Evaluation pipeline (FaceNet embeddings + nearest neighbor identification)
- Experiment notes and configuration

# What Is Not Included
Due to dataset licensing and size, this repo does not include:
- LFW images
- Cloaked image outputs
- Virtual environments

# Pipeline Overview
1. Prepare LFW face crops into identity folders
2. Generate “edited” images (JPEG + blur)
3. Generate “cloaked” images using Fawkes (TensorFlow based)
4. Evaluate recognition accuracy:
   - clean vs edited vs cloaked
   - cloaked + JPEG defense
   - cloaked + blur defense

# Metrics Reported
- Recognition accuracy (%)
- Attack success rate (drop from clean to cloaked)
- Defense recovery (how much of the lost accuracy is recovered)
- Runtime notes (practicality)

# Results

Evaluation was performed using **FaceNet embeddings** with 1-NN cosine similarity
on a subset of the **LFW dataset** (15 identities, 372 images).

## Condition & Accuracy
# Clean                | 71.79%
# Edited (JPEG / Blur) | 49.57%
# Cloaked (Fawkes)     | 62.39%
# Cloaked + JPEG       | 58.12% 
# Cloaked + Blur       | 28.21%

# Observations
- Fawkes degrades recognition accuracy but does not fully defeat modern face recognition
- Simple image space defenses (such as **Gaussian blur**) can disrupt embeddings more than cloaking alone
- Lightweight defenses provide strong privacy protection with minimal computational cost


# Quick Start
Create a Python environment, then:

```bash
pip install -r requirements.txt
python src/prepare_lfw.py
python src/make_edited.py
python src/eval_pipeline.py





# Academic Context
I built this for a graduate Secure AI course project. The goal is to evaluate how facial cloaking (Fawkes) affects recognition performance and how lightweight defenses (JPEG recompression and blur) can recover accuracy.

This repository includes the experiment code and configuration, but excludes datasets and cloaked outputs due to licensing and size constraints.
