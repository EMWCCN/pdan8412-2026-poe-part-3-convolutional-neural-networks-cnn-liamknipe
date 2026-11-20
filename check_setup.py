# Setup check for Part 3. Run it from the repo root with the venv active:  python check_setup.py
# It only reads and prints. It writes ten tiny test images to a temporary folder and deletes them at the end.
import sys
import time
import subprocess
import tempfile
import importlib

print("---")
print("Python:", sys.version.split()[0])

# I import each package on its own so one failure doesn't hide the rest
for name in ["pandas", "numpy", "matplotlib", "seaborn", "sklearn", "statsmodels", "PIL", "pyspark", "tensorflow", "keras"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: {mod.__version__}")
    except Exception as e:
        print(f"{name}: FAILED ({e})")

print("---")
# Spark 4 needs Java 17 or newer. java -version prints to stderr, not stdout.
try:
    out = subprocess.run(["java", "-version"], capture_output=True, text=True)
    # I look for the line with "version" in it, because some setups print a notice line first
    lines = [l for l in out.stderr.splitlines() if "version" in l]
    first_line = lines[0] if lines else out.stderr.strip()
    print("Java:", first_line)
    if '"17' not in first_line:
        print("Java is not 17. Spark may still work on 21, but Part 2 was set up on 17.")
except Exception as e:
    print("Java: not found on PATH", e)

print("---")
print("Tiny CNN test: 2,000 random 32x32x3 images, one epoch")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)
X = np.random.rand(2000, 32, 32, 3).astype("float32")
y = np.random.randint(0, 10, size=2000)
model = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
start = time.time()
model.fit(X, y, epochs=1, batch_size=32, verbose=0)
print("Seconds for one epoch:", round(time.time() - start, 2))
print("TensorFlow devices:", [d.device_type for d in tf.config.list_physical_devices()])

print("---")
print("Spark image reader test: ten small JPEGs")
from PIL import Image
from pyspark.sql import SparkSession

tmp = tempfile.mkdtemp()
for i in range(10):
    arr = (np.random.rand(24, 24, 3) * 255).astype("uint8")
    Image.fromarray(arr).save(f"{tmp}/test_{i}.jpg")

spark = (SparkSession.builder.master("local[*]").appName("part3_setup_check")
         .config("spark.driver.memory", "2g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
start = time.time()
imgs = spark.read.format("image").load(tmp)
imgs.printSchema()
print("Rows read:", imgs.count(), "| seconds:", round(time.time() - start, 2))
spark.stop()

import shutil
shutil.rmtree(tmp)
print("---")
print("Setup check finished")
