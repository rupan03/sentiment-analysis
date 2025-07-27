# Sentiment Analysis of User Reviews

This project presents a machine learning model designed to classify text-based user reviews into **Positive**, **Neutral**, and **Negative** sentiments. The model is built using a Logistic Regression classifier and leverages Natural Language Processing (NLP) techniques for text preprocessing and feature extraction.

The primary goal is to accurately predict sentiment from raw text, which can be invaluable for businesses analyzing customer feedback, monitoring brand reputation, and improving services.

The entire development process, from data cleaning and preprocessing to model training and evaluation, is documented in the `sentiment_analysis_project.ipynb` notebook.

***

## 🚀 Key Features

* **Text Preprocessing**: A robust pipeline that cleans text by converting to lowercase, removing punctuation and numbers, eliminating stopwords, and applying Porter Stemming.
* **Feature Extraction**: Utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert cleaned text into meaningful numerical vectors.
* **Class Imbalance Handling**: Employs **SMOTE (Synthetic Minority Over-sampling Technique)** to address the imbalanced distribution of sentiment classes in the dataset, ensuring the model doesn't get biased towards the majority class.
* **Model Training**: A **Logistic Regression** model is trained on the processed data. Hyperparameter tuning was performed using `GridSearchCV` to find the optimal settings.
* **Saved Artifacts**: The trained TF-IDF vectorizer and the final Logistic Regression model are saved as `.pkl` files for easy deployment and inference.

***

## 📊 Model Performance

The model was evaluated on a held-out test set, achieving an overall **accuracy of 79%**. The detailed classification report below shows the model's performance for each sentiment class:

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Negative** | 0.82 | 0.80 | 0.81 |
| **Neutral** | 0.32 | 0.22 | 0.26 |
| **Positive** | 0.84 | 0.92 | 0.88 |

*From the results, the model is highly effective at identifying **Positive** and **Negative** reviews. The performance on **Neutral** reviews is lower, which is a common challenge in sentiment analysis due to the subtlety and ambiguity of neutral language.*

***

## 🛠️ Tech Stack

* **Language**: Python 3
* **Libraries**:
    * Pandas & NumPy for data manipulation
    * NLTK for natural language processing tasks
    * Scikit-learn for machine learning (TF-IDF, Logistic Regression, `GridSearchCV`)
    * Imblearn for handling class imbalance (SMOTE)
    * Pickle for model serialization

***

## 📂 Repository File Structure

```
├── .gitattributes
├── PROJECT_FINAL DRAFT.csv      # The raw dataset used for training
├── README.md                    # You are here!
├── app.py                       # Python script to deploy the model (e.g., using Streamlit or Flask)
├── lr_model.pkl                 # Saved/trained Logistic Regression model
├── requirements.txt             # Required Python packages for reproducibility
├── sentiment_analysis_project.ipynb # Jupyter Notebook with all development steps
└── tfidf.pkl                    # Saved/trained TF-IDF vectorizer
```

***

## ⚙️ How to Run the Project Locally

To replicate this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
    *On Windows:*
    ```bash
    venv\Scripts\activate
    ```
    *On macOS/Linux:*
    ```bash
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(Ensure you have a `requirements.txt` file in your repository)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    To see the model in action, you can run the `app.py` script (assuming it's a Streamlit/Flask app).
    ```bash
    streamlit run app.py
    ```
    *(Note: You will need to create the `app.py` file to serve the model. It would load `lr_model.pkl` and `tfidf.pkl` to make predictions on new user input.)*
