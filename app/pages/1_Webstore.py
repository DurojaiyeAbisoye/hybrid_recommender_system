import streamlit as st
import pandas as pd
import pickle

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from recommender.recommend import recommend_top_k_items_with_details
from recommender.preprocessing import preprocess_reviews
from recommender.data_loader import load_data

# Load model and dataset
@st.cache_resource
def load_model_artifacts():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    return model, dataset

# Load data
@st.cache_data
def load_and_preprocess_data():
    reviews, metadata = load_data("data/reviews_dataset.csv", "data/metadata_dataset.csv")
    ratings, users = preprocess_reviews(reviews)
    return reviews, metadata, ratings, users

# UI setup
st.set_page_config(page_title="Recommendations", layout="wide")
st.title("🛒 Personalized Product Recommendations")

model, dataset = load_model_artifacts()
reviews_df, items_df, ratings_df, users_df = load_and_preprocess_data()

# Get user type from session
user_type = st.session_state.get("user_type", "Existing User")
st.subheader(f"Recommendations for: {user_type}")

if user_type == "Existing User":
    user_id = st.selectbox("Choose a user ID:", users_df['user_id'].unique())
    k = st.slider("Top K Recommendations", min_value=1, max_value=20, value=10)

    if st.button("Recommend"):
        st.info("Generating recommendations...")
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
        st.success("Done!")

        for _, row in rec_df.iterrows():
            image_list = eval(items_df.loc[items_df['parent_asin'] == row['item_id'], 'images'].values[0])
            image_url = image_list[0].get('hi_res') or image_list[0].get('large')

            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    st.image(image_url, width=120)
                with cols[1]:
                    st.markdown(f"**{row['title']}**")
                    category = items_df.loc[items_df['parent_asin'] == row['item_id'], 'main_category'].values[0]
                    st.markdown(f"Category: `{category}`")
                    st.markdown(f"**Score**: {row['score']:.2f}")

else:
    cold_verified = st.selectbox("Is the user a verified purchaser?", [True, False])
    k = st.slider("Top K Recommendations", min_value=1, max_value=20, value=10)

    # if st.button("Recommend (Cold Start)"):
    #     st.info("Generating recommendations...")
    #     rec_df = recommend_for_cold_start_user(
    #         model=model,
    #         dataset=dataset,
    #         user_features_dict={"verified_purchase": cold_verified},
    #         item_features=None,
    #         all_items_df=items_df,
    #         k=k
    #     )
    #     st.success("Done!")

    #     for _, row in rec_df.iterrows():
    #         image_list = eval(items_df.loc[items_df['parent_asin'] == row['item_id'], 'images'].values[0])
    #         image_url = image_list[0].get('hi_res') or image_list[0].get('large')

    #         with st.container():
    #             cols = st.columns([1, 3])
    #             with cols[0]:
    #                 st.image(image_url, width=120)
    #             with cols[1]:
    #                 st.markdown(f"**{row['title']}**")
    #                 category = items_df.loc[items_df['parent_asin'] == row['item_id'], 'main_category'].values[0]
    #                 st.markdown(f"Category: `{category}`")
    #                 st.markdown(f"**Score**: {row['score']:.2f}")
    st.warning("Cold start recommendations are not yet implemented. Please check back later.")
