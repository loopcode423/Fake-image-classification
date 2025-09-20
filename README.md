# Real vs Fake Image Classification

This repository captures the first day of experiments on spotting AI-generated (fake) images versus authentic photos using CLIP image embeddings and a simple classical classifier.

## Project Structure
- `day_1/1.ipynb` - Jupyter notebook that demonstrates the end-to-end workflow.
- `day_1/real_vs_fake_dataset/0_real` - 100 reference images labeled as real.
- `day_1/real_vs_fake_dataset/1_fake` - 100 synthetic images labeled as fake.

## Notebook Highlights
- Loads the paired real and fake image sets and visualizes representative examples.
- Uses the pretrained `openai/clip-vit-base-patch32` model to extract image feature vectors.
- Projects embeddings with t-SNE to inspect how well real and fake samples separate in 2D.
- Trains a `KNeighborsClassifier` on CLIP features and reports accuracy and a confusion matrix.

## Requirements
- Python 3.9+
- PyTorch (CPU or CUDA build)
- `transformers`
- `Pillow`
- `matplotlib`
- `numpy`
- `scikit-learn`
- `tqdm`
- `jupyterlab` or `notebook`

Install the dependencies into your environment, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow matplotlib scikit-learn tqdm jupyterlab
```

Choose the appropriate PyTorch wheel URL for your platform or GPU; the example above targets CUDA 12.1.

## Running the Notebook
1. Navigate to the project directory: `cd day_1`.
2. Launch Jupyter: `jupyter lab` (or `jupyter notebook`).
3. Open `1.ipynb` and execute the cells in order.

The notebook will download the CLIP weights on first run. A GPU is optional but accelerates feature extraction.

## Notes and Next Steps
- Double-check the feature extraction cell before training; the current notebook version reuses the real image list when initializing `fake_features`.
- Consider persisting extracted embeddings to disk if you plan to iterate quickly on downstream models.
- The closing cells contain brainstorming (in Korean) on extending CLIP plus LLM workflows; adapt as needed for your follow-up experiments.
