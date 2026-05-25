from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from common.utils import model_name

""" Load tokenizer to convert text into integer tokens """
tokenizer = AutoTokenizer.from_pretrained(model_name)

""" Pads each batch to the longest sequence at runtime """
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def df_to_dataset(df_train, df_validation, df_test):
    """ Convert dataframes into HuggingFace datasets """
    ds_train = Dataset.from_pandas(df_train[['text', 'label']])
    ds_val = Dataset.from_pandas(df_validation[['text', 'label']])
    ds_test = Dataset.from_pandas(df_test[['text', 'label']])
    return ds_train, ds_val, ds_test


def tokenize(batch):
    """ Tokenize a batch of texts """
    # Get token IDs: take batch from 'text' column, truncate if more than max_length
    tokenized_inputs = tokenizer(batch["text"],
                                 truncation=True,
                                 padding=False,   # padding is handled by data collator
                                 max_length=128)

    # Ensure that sentiment_label and emotion_label are passed through if they exist in the batch
    if "sentiment_label" in batch:
        tokenized_inputs["sentiment_label"] = batch["sentiment_label"]
    if "emotion_label" in batch:
        tokenized_inputs["emotion_label"] = batch["emotion_label"]
    return tokenized_inputs


def tokenize_datasets(ds_train, ds_val, ds_test):
    """ Apply tokenize function to each dataset split """
    ds_train = ds_train.map(tokenize, batched=True)
    ds_val = ds_val.map(tokenize, batched=True)
    ds_test = ds_test.map(tokenize, batched=True)
    return ds_train, ds_val, ds_test
