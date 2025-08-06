import pandas as pd

def load_data(reviews_path, metadata_path):
    reviews = pd.read_csv(reviews_path)
    metadata = pd.read_csv(metadata_path)

    common_asins = set(reviews['parent_asin']) & set(metadata['parent_asin'])
    reviews = reviews[reviews['parent_asin'].isin(common_asins)]
    metadata = metadata[metadata['parent_asin'].isin(common_asins)]

    return reviews, metadata
