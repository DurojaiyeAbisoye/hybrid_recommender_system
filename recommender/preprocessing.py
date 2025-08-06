import pandas as pd

def preprocess_reviews(reviews):
    ratings = reviews[['user_id', 'parent_asin', 'rating', 'helpful_vote']].rename(columns={'parent_asin': 'item_id'})
    users = reviews[['user_id', 'verified_purchase']]
    return ratings, users

def preprocess_items(metadata):
    items = metadata[['parent_asin', 'main_category', 'price']]
    items = items.rename(columns={'parent_asin': 'item_id'})
    items = pd.get_dummies(items, columns=['main_category'], prefix='main_category', dtype=int)
    
    items['price_bin'] = pd.qcut(items['price'], 10, labels=False)
    
    items = pd.get_dummies(items, columns=['price_bin'], dtype=int)
    items.drop(columns=['price'], inplace=True)
    return items

def encode_user_features(users):
    users = pd.get_dummies(users, columns=['verified_purchase'], prefix='verified_purchase', dtype=int)
    user_feats = users.drop(columns=['user_id']).to_dict(orient='records')
    return users, user_feats

def encode_item_features(items):
    item_feats = items.drop(columns=['item_id']).to_dict(orient='records')
    return items, item_feats
