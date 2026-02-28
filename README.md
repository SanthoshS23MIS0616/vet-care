# CanineCare - AI Vet For Dog Breeds

This repository explores using computer vision and machine learning to improve dog care. It includes tools for identifying dog breeds from images, predicting diseases, and recommending the best breed for potential owners. By analyzing images and data, these tools help veterinarians provide better care and help people choose the right dog for their lifestyle, ensuring healthier and happier dogs and owners.

# Canine Care and Management System

## Overview

This project leverages computer vision and machine learning to improve dog care through three main functionalities: dog breed classification, disease prediction, and breed recommendation. The application is built using Flask and integrates TensorFlow and Scikit-learn models to deliver these functionalities.

## Features

1. **Dog Breed Classification**
    - Identifies dog breeds from images using a pre-trained TensorFlow model.
2. **Disease Prediction**
    - Predicts potential diseases based on symptoms using a Scikit-learn model.
3. **Breed Recommendation System**
    - Recommends dog breeds suitable for different lifestyles and preferences.

## Requirements

- Flask
- TensorFlow
- TensorFlow Hub
- Scikit-learn
- Joblib
- OpenCV
- NumPy
- Pandas

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ssbaraar/CanineCare-AI-Vet-for-Dog-Breeds.git
    cd CanineCare-AI-Vet-for-Dog-Breeds
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have the required model files and place them in the `model/` directory:
    - TensorFlow model: `model/20220804-16551659632113-all-images-Adam.h5`
    - Scikit-learn model: `model/dogModel1.pkl`

5. Ensure you have the required data file and place it in the `data/` directory:
    - Data file: `data/dog_data_09032022.csv`

## Running the Application

1. Start the Flask application:
    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Screenshots

### Home Page

![Home Page](sample/HomePage.png)

![Services Page](sample/Services.png)

### Dog Breed Classification Page

![Classification Page](sample/classify.png)

![Classification results of Dog Breed](/sample/resultsc.png)

### Dog Breed Recommendation Page

![Dog Recommendation Page](/sample/recomm.png)

![Results](/sample/resultsr.png)


### Predict Disease Page
![Predict Disease](sample/disease.png)

![Results](/sample/resultsp.png)

## Endpoints

### Home
- **URL:** `/`
- **Methods:** `GET`, `POST`
- **Description:** Renders the home page where users can upload an image to predict the dog breed.

### Predict Dog Breed
- **URL:** `/predict_breed_route`
- **Methods:** `POST`
- **Description:** Accepts an image file and returns the predicted dog breed.
- **Request:**
    - `file`: Image file of the dog.
- **Response:**
    ```json
    {
        "predicted_breed": "Golden Retriever"
    }
    ```

### Predict Disease
- **URL:** `/predict_disease`
- **Methods:** `POST`
- **Description:** Accepts symptoms data and returns the predicted disease.
- **Request:**
    ```json
    {
        "symptoms": [0.1, 0.3, 0.5, ...]
    }
    ```
- **Response:**
    ```json
    {
        "disease": "Tick fever"
    }
    ```

## File Structure
```
│
├── app.py # Main Flask application
├── model/
│   ├── 20220804-16551659632113-all-images-Adam.h5 # TensorFlow model
│   └── dogModel1.pkl # Scikit-learn model
├── data/
│   └── dog_data_09032022.csv # Data file for preprocessing
├── templates/
│   └── Home.html # HTML template for the home page
├── static/
│   └── style.css # CSS file for styling
├── requirements.txt # List of Python dependencies
└── README.md # Project documentation
```


