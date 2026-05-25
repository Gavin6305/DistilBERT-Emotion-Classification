import torch
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Local imports
from models.flat import load_flat_model
from data.dataset import tokenizer, data_collator
from common.utils import get_device


def compute_metrics(pred):
    """ Computes accuracy and f1_score given predictions and labels """
    logits = pred.predictions
    preds = logits.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


def train_flat(ds_train,
               ds_val,
               output_dir="./flat_cls",
               eval_strategy="epoch",
               save_strategy="epoch",
               save_total_limit=1,
               learning_rate=2e-5,
               per_device_train_batch_size=16,
               per_device_eval_batch_size=16,
               num_train_epochs=5,
               weight_decay=0.01,
               load_best_model_at_end=True,
               metric_for_best_model="eval_loss",
               greater_is_better=False,
               logging_steps=50):
    """
    Trains the flat DistilBERT model

    Args:
        ds_train:                       training dataset
        ds_val:                         validation dataset
        output_dir:                     directory to save the model
        eval_strategy:                  how often to run evaluation
        save_strategy:                  how often to save the model
        save_total_limit:               limit saving to first best epoch
        learning_rate:                  base learning rate for the AdamW optimizer
        per_device_train_batch_size:    training batch size per device
        per_device_eval_batch_size:     evaluation batch size per device
        num_train_epochs:               number of training epochs
        weight_decay:                   L2 weight decay regularization for better generalization
        load_best_model_at_end:         after training, reload the checkpoint with best eval score
        metric_for_best_model:          metric used to decide which checkpoint is “best”
        greater_is_better:              less val loss is better
        logging_steps:                  log metrics (loss, lr, etc.) every 50 training steps
    """
    # Define HF training args
    training_args = TrainingArguments(output_dir=output_dir,
                                      eval_strategy=eval_strategy,
                                      save_strategy=save_strategy,
                                      save_total_limit=save_total_limit,
                                      learning_rate=learning_rate,
                                      per_device_train_batch_size=per_device_train_batch_size,
                                      per_device_eval_batch_size=per_device_eval_batch_size,
                                      num_train_epochs=num_train_epochs,
                                      weight_decay=weight_decay,
                                      load_best_model_at_end=load_best_model_at_end,
                                      metric_for_best_model=metric_for_best_model,
                                      greater_is_better=greater_is_better,
                                      logging_steps=logging_steps)
    # Load model
    model = load_flat_model()

    # Define HF trainer object
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=ds_train,
                      eval_dataset=ds_val,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics)

    # Train the model
    trainer.train()


def test_flat(model_path="./flat_cls/checkpoint-5428",  ds_test=None):
    """ Tests a flat model on ds_test"""
    # Load model and tokenizer (I know tokenizer here shadows, but they are both the same exact thing)
    model, tokenizer = load_flat_model(model_path)

    # Create a pytorch DataLoader
    ds_test_tensor = ds_test.remove_columns(["text"])
    test_loader = DataLoader(ds_test_tensor,
                             batch_size=16,
                             shuffle=False,
                             collate_fn=data_collator)

    # Get device
    device = get_device()

    # Run inference manually
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Macro F1:", f1_score(all_labels, all_preds, average="macro"))


