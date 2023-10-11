import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.utils import pad_sequences


idx2category = {
    0: "Business",
    1: "Entertainment",
    2: "Politics",
    3: "Sports",
    4: "Technology"
}


def get_category(predictions: np.ndarray) -> str:
    """
    Get the category of the maximum score in `predictions`.

    Parameters:
    ------------
    - predictions : np.ndarray
        Contains 5 scores repesenting the predcited score for each category.

    Returns:
    --------
    - str
        Represents the predicted score using `idx2category` dictionary.
    """
    predicted_index = np.argmax(predictions)
    return idx2category[predicted_index]


class LstmClassifier:
    """Responsible for make predictions on a text using LSTM-based model"""

    def __init__(self) -> None:
        self.lstm_model = keras.models.load_model(
            "F:/BBC-Text-Classification/models/BBC-Classifier-LSTM.h5")
        with open("F:/BBC-Text-Classification/tokenizers/lstm_tokenizer.pickle", 'rb') as handle:
            self.lstm_tokenizer = pickle.load(handle)

    def convert_text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert passed `text` to a numerical vector.

        Parameters:
        ------------
        - text : str
            Text needed to be converted to vector.

        Returns:
        --------
        - text_vector_padded : np.ndarray
            A vector representation of size 256 of `text`.
        """
        text = pd.Series(text)
        text_vector = self.lstm_tokenizer.texts_to_sequences(text)
        text_vector_padded = pad_sequences(
            text_vector, maxlen=256, padding="post", truncating="post")

        return text_vector_padded

    def predict(self, text_vector: np.ndarray) -> np.ndarray:
        """
        Get predictions scores of the passes `text_vector`

        Parameters:
        ------------
        - text_vector : np.ndarray
            A vector representation of a text of size 256

        Returns:
        --------
        - predictions : np.ndarray
            A vector of size number of categories "5" represents
            the score of each category.
        """
        predictions = self.lstm_model.predict(text_vector)
        return predictions


class GruClassifier:
    """Responsible for make predictions on a text using GRU-based model"""

    def __init__(self) -> None:
        self.gru_model = keras.models.load_model(
            "F:/BBC-Text-Classification/models/BBC-Classifier-GRU.h5")
        with open("F:/BBC-Text-Classification/tokenizers/gru_tokenizer.pickle", 'rb') as handle:
            self.gru_tokenizer = pickle.load(handle)

    def convert_text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert passed `text` to a numerical vector.

        Parameters:
        ------------
        - text : str
            Text needed to be converted to vector.

        Returns:
        --------
        - text_vector_padded : np.ndarray
            A vector representation of size 256 of `text`.
        """
        text = pd.Series(text)
        text_vector = self.gru_tokenizer.texts_to_sequences(text)
        text_vector_padded = pad_sequences(
            text_vector, maxlen=256, padding="post", truncating="post")

        return text_vector_padded

    def predict(self, text_vector: np.ndarray) -> np.ndarray:
        """
        Get predictions scores of the passes `text_vector`

        Parameters:
        ------------
        - text_vector : np.ndarray
            A vector representation of a text of size 256

        Returns:
        --------
        - predictions : np.ndarray
            A vector of size number of categories "5" represents
            the score of each category.
        """
        predictions = self.gru_model.predict(text_vector)
        return predictions
