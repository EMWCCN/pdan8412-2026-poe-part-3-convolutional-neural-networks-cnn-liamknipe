# Data

The images are not stored in this repository (about 760 MB).

Dataset: The Garbage Dataset (GD), S. Kunwar, 2026. 12,259 images in ten classes.
Download: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
Paper: S. Kunwar, "The Garbage Dataset (GD): A multi-class image benchmark for automated waste segregation," J. Braz. Comput. Soc., vol. 32, no. 1, 2026, doi: 10.5753/jbcs.2026.7774.

After downloading, this folder should contain two of the dataset's folders, each with ten class folders inside:

```
data/
├── original/            the original images (used for size, brightness and file-size metadata)
└── standardized_256/    the 256 x 256 padded images (used for training)
```

The dataset's third folder (`standardized_384`) is not needed.
