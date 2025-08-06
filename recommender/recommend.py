import numpy as np
import pandas as pd

def recommend_top_k_items_with_details(
    model,
    user_id,
    dataset,
    user_features,
    item_features,
    all_items_df,
    ratings_df,
    k=10
):
    user_index = dataset.mapping()[0][user_id]
    item_id_to_index = dataset.mapping()[2]
    index_to_item_id = {v: k for k, v in item_id_to_index.items()}

    known_item_ids = ratings_df[ratings_df['user_id'] == user_id]['item_id'].tolist()
    known_indices = [item_id_to_index[i] for i in known_item_ids if i in item_id_to_index]

    scores = model.predict(
        user_ids=user_index,
        item_ids=np.arange(len(item_id_to_index)),
        user_features=user_features,
        item_features=item_features
    )
    scores[known_indices] = -np.inf

    top_indices = np.argsort(-scores)[:k]
    data = []
    for i in top_indices:
        item_id = index_to_item_id[i]
        title = all_items_df.loc[all_items_df['parent_asin'] == item_id, 'title'].values
        title = title[0] if len(title) else "Unknown Title"
        data.append({'item_id': item_id, 'title': title, 'score': scores[i]})

    return pd.DataFrame(data).sort_values(by='score', ascending=False).reset_index(drop=True)