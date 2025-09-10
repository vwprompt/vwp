# Visual Writing Prompts (VWP)

**[Hugging Face Datasets (New!)](https://huggingface.co/datasets/tonyhong/vwp)** | **[Github Repository](https://github.com/vwprompt/vwp)** | **[arXiv e-Print (colorful figures)](https://arxiv.org/abs/2301.08571)** | [TACL 2023 (black&white figures)](https://doi.org/10.1162/tacl_a_00553) | **[Website](https://vwprompt.github.io/)**

> This repository accompanies the paper *Visual Writing Prompts: Character-Grounded Story Generation with Curated Image Sequences* by Xudong Hong, Asad Sayeed, Khushboo Mehra, Vera Demberg, and Bernt Schiele. It provides code and pointers to the VWP dataset for training and evaluating character-grounded visual story generation models. 

---

## Overview

Visual story generation often struggles because existing image sequences don’t naturally support coherent plots. VWP fixes this by curating image sequences from movies and explicitly grounding recurring characters, which leads to stories that are more coherent, diverse, and visually grounded. The paper also introduces a **character-grid Transformer (CharGrid)** baseline that models visual coherence via character continuity and outperforms prior work (e.g., TAPM). 

**What you’ll find here**

- Preprocessing and model training code for character-grounded story generation.
- Scripts/utilities to work with the VWP dataset.

---

## Dataset

The Visual Writing Prompts (VWP) dataset contains almost 2K selected sequences of movie shots, each including 5-10 images. The image sequences are aligned with a total of 12K stories, which are collected via crowdsourcing given the image sequences and up to 5  grounded characters from the corresponding image sequence.

### Load with 🤗 Datasets

```python
from datasets import load_dataset

# Public dataset ID on Hugging Face
ds = load_dataset("tonyhong/vwp")

print(ds)              # splits: train / val / test
print(ds["train"][0])  # fields include 'story', 'sep_story', 'anonymised_story',
                       # 'img_id_list', 'char0..char4', 'char*_url', 'link*', etc.
```

> Notes:
> - `story` is the full free-form story; `sep_story` inserts `[SENT]` to mark image-aligned segments; `anonymised_story` replaces named entities with placeholders like `[male0]`. 

---

## Installation

We recommend Python 3.9+.

```bash
# create environment (example with venv)
python -m venv .venv && source .venv/bin/activate

# install core deps
pip install -r requirements.txt
# or, if you prefer to install manually:
pip install torch torchvision transformers timm datasets pillow tqdm
```

---

## Models & Features

The repository includes the paper’s **CharGrid** baseline and utilities to train/evaluate it alongside GPT-2–based baselines. Visual inputs use **Swin Transformer** features (global), **Cascade Mask R-CNN** detections (objects), and **character crops** derived from VWP’s bounding boxes; the character-grid encodes character-image similarity over the sequence to model coherence. Text generation uses a GPT-2 decoder. 

---

## Results (high-level)

CharGrid produces stories that are **more coherent, visually grounded, and diverse** than those from the state-of-the-art TAPM model on VWP, according to both automatic metrics (BLEU-1/2/3/4, METEOR, ROUGE-L, and a coherence proxy) and human judgments. See the paper for complete tables and annotation details. 

---

## Repository Structure
```
.
├── code/ # data prep, feature extraction, training, eval
├── column_explain.csv # column descriptions for dataset rows
```

---

## Citing

If you use VWP or the CharGrid baseline, please cite the paper:

```bibtex
@article{hong-etal-2023-visual-writing,
  title   = {Visual Writing Prompts: Character-Grounded Story Generation with Curated Image Sequences},
  author  = {Hong, Xudong and Sayeed, Asad and Mehra, Khushboo and Demberg, Vera and Schiele, Bernt},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {11},
  pages   = {565--581},
  year    = {2023},
  doi     = {10.1162/tacl_a_00553},
  url     = {https://aclanthology.org/2023.tacl-1.33/}
}
```
_BibTeX source: ACL Anthology._


## Acknowledgements

Image sequences were curated from **MovieNet**, and the model leverages **Swin Transformer** features and **Cascade Mask R-CNN** detections as described in the paper. We thank all annotators who contributed stories via AMT. 

---

## FAQ

- **Where do I get the dataset images?** Use the Hugging Face dataset; records include image IDs/URLs for the curated sequences and character crops along with stories.   
- **How many images per sequence?** 5–10 curated frames per sequence. 

---

## Contact

Xudong Hong

xLASTNAME@coli.uni-saarland.de

Found a bug or have a question? Please open an issue in this repository. (For scholarly questions, see the paper’s author information.) 

*Happy storytelling!*

---

# Disclaimer:

All the images are extracted from the movie shots from the MovieNet dataset (https://opendatalab.com/OpenDataLab/MovieNet/tree/main/raw). The copyrights of all movie shots belong to the original copyright holders, which can be found on the IMDb page of each movie. The IMDb page is indicated by the index in the `imdb_id` column. For example, for the first row of our data, the `imdb_id` is `tt0112573`, so the corresponding IMDb page is https://www.imdb.com/title/tt0112573/companycredits/. Do not violate copyrights while using these images. We only use these images for academic purposes. Please contact the author if you have any questions.

---
