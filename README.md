# BBC News Classification NLP Application
This project is an NLP (Natural Language Processing) application that classifies BBC news articles into different genres, including sports, politics, entertainment, business, and technology. The classification is done using two different techniques: LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit). The models achieve an accuracy of approximately 94%.

**Table of Contents:**
* [Introduction](#introduction)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Usage](#usage)
* [Demo](#demo)

# Introduction
In this project, we aim to classify BBC news articles into different genres using NLP techniques. The LSTM and GRU models are trained on a labeled dataset to learn the patterns and features of each genre. The models are then used to predict the genre of new, unseen news articles.

# Technologies Used
- Python
- re
- NumPy
- Pandas
- Scikit-Learn
- TensorFlow
- Keras
- HuggingFace transformers
- Streamlit

# Installation
To run this project locally, follow these steps:

1. Clone the repository:
```
git clone https://github.com/MohammedAly22/TextClassifier
```

2. Run the application:
```
streamlit run src/app.py
```

**Note**: Make sure that you have installed all the packages required for this project.

# Usage
Once the application is running, you can access it through your web browser. Enter the text of a BBC news article, and the application will classify it into one of the genres: sports, politics, entertainment, business, or technology.

# Demo
https://github.com/MohammedAly22/TextClassifier/assets/90681796/ba747708-25bd-40af-8df8-da5d019ba18d


