# PDAN8412 POE Part 3: Sorting waste images with a convolutional neural network

**Student:** Liam Knipe (ST10355049)
**Module:** PDAN8412 Programming for Data Analytics 2
**Part:** 3 of 3 (image recognition with TensorFlow)

## What this is

A notebook that trains a CNN from scratch to sort photos of single waste items into ten material classes (battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash). The scenario is a fictional book press that wants image recognition at its premises; a sorting point is the first use. The notebook compares the CNN with three baselines, including a "shortcut" model that only sees image size, brightness and file size, and tests it under darker, brighter, blurred, rotated and compressed conditions.

## Which library does what

| Task | Tool |
|---|---|
| Reading all images, metadata, quality checks, class aggregates | PySpark (`binaryFile` reader with a pandas UDF, and the image reader) |
| Loading pixels for training (after the marked Spark hand-off) | Pillow and NumPy |
| CNN, dense baseline, training | TensorFlow / Keras |
| Shortcut baseline, metrics, class weights | scikit-learn |
| McNemar's test | statsmodels |

## Repository layout

```
.
├── README.md
├── requirements.txt       pinned package versions
├── check_setup.py         checks Python, Java, the packages, a tiny CNN and Spark's image reader
├── data/                  the images go here (see data/README.md); not in the repo
├── notebooks/
│   └── PDAN8412_Part3_ST10355049_LiamKnipe.ipynb
├── artifacts/             saved models, training histories, predictions, split, summary tables
├── figures/               figures saved by the notebook (used in the report)
└── report/
    └── PDAN8412_Part3_Report_ST10355049_LiamKnipe.pdf
```

The notebook uses relative paths (`../data/`, `../artifacts/`, `../figures/`), so it must be run from inside `notebooks/` (Jupyter and VS Code both do this by default).

## Setup

I developed and ran this on macOS (Apple M2, 8 GB) with **Python 3.13.11** and **Java 17**, on the CPU. I have not run it on Windows.

1. **Python 3.13** and **Java 17 or newer** are needed. PySpark runs on the JVM, so `java -version` must work in the terminal you start Jupyter or VS Code from.
2. Create an environment and install the packages:

   ```
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   With Anaconda, create a Python 3.13 environment instead and run `pip install -r requirements.txt` inside it.
3. Download the data into `data/` as described in `data/README.md`.
4. Check the setup: `python check_setup.py`
5. Open `notebooks/PDAN8412_Part3_ST10355049_LiamKnipe.ipynb`, choose the environment from step 2 as the kernel, and run **Restart and Run All**.

## Run time and the saved models

The trained models are in `artifacts/` (`dense_baseline.keras`, `cnn_v1.keras`, `cnn_v2.keras`). With `RETRAIN = False` (the default, in the settings cell) the notebook loads them instead of training, so a full run takes a few minutes, mostly Spark reading the 12,259 images. Setting `RETRAIN = True` trains all three again; on my laptop that took about two hours (CNN v1 44.5 minutes, CNN v2 63.3 minutes). Spark is set to 2 GB of driver memory and 8 shuffle partitions so it fits on an 8 GB laptop. The seed is 42.

## Results in one line

On the test set the class-weighted CNN reaches a macro F1 of 0.590 (95% interval 0.567 to 0.612), against 0.456 for the shortcut model and 0.316 for the dense network. The report explains what that does and does not mean for the press.
