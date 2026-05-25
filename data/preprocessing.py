import pandas as pd
from common.labels import *


def load_parquet_splits ():
    """Get data splits"""
    go_emotions_link = "hf://datasets/google-research-datasets/go_emotions/"

    splits = {'train': 'simplified/train-00000-of-00001.parquet',
              'validation': 'simplified/validation-00000-of-00001.parquet',
              'test': 'simplified/test-00000-of-00001.parquet'}

    df_train = pd.read_parquet(go_emotions_link + splits["train"])
    df_validation = pd.read_parquet(go_emotions_link + splits["validation"])
    df_test = pd.read_parquet(go_emotions_link + splits["test"])

    return df_train, df_validation, df_test


def preprocess_splits (df_train, df_validation, df_test):
    """ Pre-processes splits so that they:
            - Have 1 label per emotion
            - Have original emotion, sentiment and ekman label columns
    """
    # If rows have multiple labels, choose the first
    for df in [df_train, df_validation, df_test]:
        df['label'] = df['labels'].apply(lambda x: x[0])
        df.drop(columns=['labels'], inplace=True)

    # Make sentiment and ekman label columns
    for df in [df_train, df_validation, df_test]:
        df['sentiment_label'] = df['label'].apply(lambda x: inv_sentiment_dict[actual_labels[x]])
        df['ekman_label'] = df['label'].apply(lambda x: inv_ekman_dict[actual_labels[x]])

    return df_train, df_validation, df_test
