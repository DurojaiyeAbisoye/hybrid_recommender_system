# Hybrid Recommender System

![Streamlit](https://streamlit.io/images/brand/streamlit-mark-color.png)

This project implements a hybrid recommender system using [LightFM](https://making.lyst.com/lightfm/docs/home.html), combining collaborative and content-based filtering for personalized product recommendations. The system is designed to train on user reviews and product metadata, and provides an interactive web interface for exploring recommendations.

## Features

- **Hybrid Recommendation**: Leverages both user-item interactions and item/user features.
- **Streamlit Web App**: User-friendly interface for exploring recommendations.
- **Customizable Training Pipeline**: Easily retrain the model with new data.
- **Efficient Model Storage**: Trained models and datasets are saved for fast loading.

## Project Structure
```
├── main.py  # Training pipeline script
├── app/                     # Streamlit web app
│   ├── Home.py
│   └── pages/
│       └── 1_Webstore.py
├── recommender/             # Core recommender logic
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── recommend.py
│   └── train_model.py
├── data/                    # Datasets (CSV files)
    ├── metadata_dataset.csv
    ├── reviews_dataset.csv
├── models/                  # Saved model and dataset
├── requirements.txt         # Python dependencies
└── README.md
```

## 🛠️ How to Run Locally
```sh
git clone https://github.com/DurojaiyeAbisoye/hybrid_recommender_system.git
cd hybrid_recommender_system
pip install -r requirements.txt
 streamlit run app/Home.py
```
fastapi_implementation
2. Prepare Data
Place your data files in the data/ directory:

reviews_dataset.csv: User reviews and ratings.
metadata_dataset.csv: Product metadata.
3. Train the Model
Run the training pipeline:

python [main.py]

4. Launch the Web App
Start the Streamlit app:

To run the fastapi server:
```
uvicorn serve:app --reload
```

