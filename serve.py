from fastapi import FastAPI
import pandas as pd
import pickle
import sys
import os
from contextlib import asynccontextmanager


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from recommender.recommend import recommend_top_k_items_with_details
from recommender.preprocessing import preprocess_reviews
from recommender.data_loader import load_data

# Load model and dataset
def load_model_artifacts():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    return model, dataset


def load_and_preprocess_data():
    reviews, metadata = load_data("data/reviews_dataset.csv", "data/metadata_dataset.csv")
    ratings, users = preprocess_reviews(reviews)
    return reviews, metadata, ratings, users


async def lifespan(app: FastAPI):
    app.state.model, app.state.dataset = load_model_artifacts()
    app.state.reviews, app.state.metadata, app.state.ratings, app.state.users = load_and_preprocess_data() 
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/old_user_recommendations")
def get_old_user_recommendations(user_id: str, k: int = 10):
    model = app.state.model
    dataset = app.state.dataset
    
    reviews_df, items_df, ratings_df, users_df = (
        app.state.reviews, app.state.metadata, app.state.ratings, app.state.users)
    rec_df = recommend_top_k_items_with_details(
        model=model,
        user_id=user_id,
        dataset=dataset,
        user_features=None,
        item_features=None,
        all_items_df=items_df,
        ratings_df=ratings_df,
        k=k
    )
    return rec_df.to_dict(orient='records')


@app.get("/new_user_recommendations")
def get_new_user_recommendations(user_id: int = 0, k: int = 10):
    model = app.state.model
    dataset = app.state.dataset
    
    reviews_df, items_df, ratings_df, users_df = (
        app.state.reviews, app.state.metadata, app.state.ratings, app.state.users)
    rec_df = recommend_top_k_items_with_details(
        model=model,
        user_id=user_id,
        dataset=dataset,
        user_features=None,
        item_features=None,
        all_items_df=items_df,
        ratings_df=ratings_df,
        k=k
    )
    return rec_df.to_dict(orient='records')
