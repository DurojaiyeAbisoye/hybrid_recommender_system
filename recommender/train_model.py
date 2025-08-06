import pickle
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import coo_matrix, csr_matrix

def build_dataset(users, items, user_features_col, item_features_col):
    dataset = Dataset()
    dataset.fit(users=users['user_id'].unique(),
                items=items['item_id'].unique(),
                user_features=user_features_col,
                item_features=item_features_col)
    return dataset

def build_features(dataset, users, user_feats, items, item_feats):
    user_features = dataset.build_user_features((x, y) for x, y in zip(users['user_id'], user_feats))
    item_features = dataset.build_item_features((x, y) for x, y in zip(items['item_id'], item_feats))
    return user_features, item_features

def build_interactions(dataset, ratings):
    ratings['liked'] = ratings['rating'].apply(lambda x: 1 if x >= 3 else 0)
    interactions, weights = dataset.build_interactions(
        (x, y, z) for x, y, z in zip(ratings['user_id'], ratings['item_id'], ratings['liked'])
    )
    return interactions, weights

def clean_duplicates(matrix):
    if not isinstance(matrix, coo_matrix):
        matrix = matrix.tocoo()
    matrix = matrix.tocsr().tocoo()
    return matrix

def train_model(train, train_w, user_features, item_features, n_components=30, loss='warp', epochs=30, num_threads=1):
    model = LightFM(no_components=n_components, loss=loss, random_state=1616)
    model.fit(train, sample_weight=train_w, user_features=user_features, item_features=item_features,
              epochs=epochs, num_threads=num_threads)
    return model

def save_model(model, dataset, path='models/'):
    with open(f'{path}/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{path}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
