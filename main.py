# main.py
import os
import logging
import numpy as np
from recommender.data_loader import load_data
from recommender.preprocessing import (
    preprocess_reviews, preprocess_items,
    encode_user_features, encode_item_features
)
from recommender.train_model import (
    build_dataset, build_features, build_interactions,
    clean_duplicates, train_model, save_model, save_features
)
from lightfm.cross_validation import random_train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("🚀 Starting training pipeline...")

    # Load raw data
    logging.info("📥 Loading raw data...")
    reviews, metadata = load_data("data/reviews_dataset.csv", "data/metadata_dataset.csv")

    # Preprocess
    logging.info("🧹 Preprocessing reviews and items...")
    ratings, users = preprocess_reviews(reviews)
    items = preprocess_items(metadata)

    logging.info("🔢 Encoding features...")
    users, user_feats = encode_user_features(users)
    items, item_feats = encode_item_features(items)

    # Build dataset
    logging.info("🧱 Building LightFM dataset and features...")
    user_features_col = users.drop(columns=['user_id']).columns.values
    item_features_col = items.drop(columns=['item_id']).columns.values
    dataset = build_dataset(users, items, user_features_col, item_features_col)
    user_features, item_features = build_features(dataset, users, user_feats, items, item_feats)

    # Save features
    logging.info("💾 Saving user and item features...")
    os.makedirs("models", exist_ok=True)
    save_features(user_features, item_features, path="models")
    # Build interactions
    logging.info("🔗 Creating interaction and weight matrices...")
    interactions, weights = build_interactions(dataset, ratings)
    interactions = clean_duplicates(interactions)
    weights = clean_duplicates(weights)

    # Train-test split
    logging.info("✂️ Splitting into train and test sets...")
    train, test = random_train_test_split(interactions, test_percentage=0.2, random_state=np.random.RandomState(42))
    train_w, test_w = random_train_test_split(weights, test_percentage=0.2, random_state=np.random.RandomState(42))

    # Train model
    logging.info("🏋️ Training model...")
    model = train_model(train, train_w, user_features, item_features)

    # Save model
    logging.info("💾 Saving model and dataset...")
    os.makedirs("models", exist_ok=True)
    save_model(model, dataset, path="models")

    logging.info("✅ Training pipeline complete!")

if __name__ == "__main__":
    main()


