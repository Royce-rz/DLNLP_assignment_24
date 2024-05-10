# DLNLP_assignment_24
## Project Description
This is the assignment for ELEC0141 Deep Learning for Natural Language Processing.This project focuses on leveraging deep learning techniques for sentiment analysis, specifically targeting movie reviews from the IMDb dataset. It employs advanced natural language processing methods, including GloVe embeddings for transforming text into meaningful vector spaces and LSTM networks to effectively understand and process the temporal dependencies in text. The objective is to develop a robust model capable of accurately classifying sentiments expressed in movie reviews as either positive or negative.


## Data Preperation
The data is to large that can not upload to Github,The following link is to download the data.


IMDB Dataset of 50K Movie Reviews: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews


Pickled glove.840B.300d: https://www.kaggle.com/datasets/authman/pickled-glove840b300d-for-10sec-loading


### Requirement

This step series is to show how to get a development environment running for this project.

1. **Clone the repository**

   ```bash
   git clone https://yourrepository.git
   ```
2.**create the environment and run**
   ```bash
   conda env create -f environment.yml
   python main.py
   ```



## Training Details

### Hardware Used

- **Chip**: Apple M2
- **GPU Acceleration**: Enabled
- **Training Environment**: TensorFlow with GPU support

### Training Time

The model was trained on an Apple M2 chip with GPU acceleration, resulting in a total training time of approximately 7000 seconds. This efficient use of hardware demonstrates the model's capability to handle extensive computations and large datasets effectively.

## Output IMDb Sentiment Analysis Model

This project develops a deep learning model to perform sentiment analysis on IMDb movie reviews. The goal is to classify reviews as positive or negative accurately.


The output of this project is a trained model saved as `imdb_model.h5`. This file contains the architecture of the model along with its trained weights.

### What is `imdb_model.h5`?

The `imdb_model.h5` file is a HDF5 file that stores:
- The model's architecture (layers, activations, etc.)
- The weights of the model after training
- The training configuration (loss, optimizer)
- The state of the optimizer, allowing to resume training exactly where you left off.

## How to Use the Model

### Prerequisites

Ensure you have the following installed:
- Python 3.6 or higher
- TensorFlow 2.x
- h5py (for handling `.h5` files)

### Loading the Model

You can load the model using TensorFlow Keras API:

```python
from tensorflow.keras.models import load_model

# Load the model
model = load_model('imdb_model.h5')

# Use the model to make predictions
# `sample_text` should be preprocessed as per the model's training data
prediction = model.predict(sample_text)
print("Prediction:", prediction)

