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
1. Install `virtualenv` (if not installed):
```
pip install virtualenv
```

2. Create a new virtual environment with your preferred name, here I called it `bbc-news-classification-venv`:
```
virtualenv bbc-news-classification-venv
```

3. Activate the created virtual environment using this command for Windows users:
```
bbc-news-classification-venv\Scripts\activate
```

4. Clone the repository in the same directory that your created virtual environment inside:
```
git clone https://github.com/MohammedAly22/BBCNewsClassifier
```

5. install the project requirements:
```
pip install -r requirements.txt
```

6. Run the application:
```
streamlit run src/app.py
```

7. Don't forget to deactivate the virtual environment after that:
```
deactivate
```

# Usage
Once the application is running, you can access it through your web browser. Enter the text of a BBC news article, and the application will classify it into one of the genres: sports, politics, entertainment, business, or technology.

# Demo
https://github.com/MohammedAly22/BBCNewsClassifier/assets/90681796/ba747708-25bd-40af-8df8-da5d019ba18d


