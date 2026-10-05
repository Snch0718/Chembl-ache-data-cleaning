# AChE Bioactivity Data Preparation & Optimization Pipeline

A Python-based bioactivity data processing pipeline for filtering, cleaning, deduplicating, and feature-engineering Acetylcholinesterase (AChE) inhibitor datasets obtained from ChEMBL.

## Features

- **Data Cleaning**: Drops empty columns, filters specifically for exact (`=`) IC50 measurements in `nM` units, and removes invalid records.
- **Deduplication**: Handles duplicate SMILES compounds by computing the median potency values across assays.
- **Feature Engineering**: Calculates biological potency metrics (`pIC50`).
- **Molecular Descriptors**: Uses RDKit to parse canonical SMILES and generate 13 molecular property descriptors (e.g., MW, LogP, TPSA, HBD, HBA).
- **Synthetic Accessibility**: Estimates compound synthetic accessibility (SA) score heuristics.
- **Data Visualization**: Saves distribution plots, descriptor correlation heatmaps, and potency vs. SA scatter plots.

## Prerequisites

Ensure you have Python 3.9+ installed.

### Installation

Clone the repository and install required dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt