from transformers import AutoModelForSequenceClassification, AutoTokenizer
from common.utils import model_name, get_device


def load_flat_model(model_path=None):
    """ Loads the uncased DistilBERT either fresh or from a given path (loading from path will put model in eval). """
    # Fresh with 28 label classification head
    if model_path is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=28)
        return model

    # Load model and tokenizer from a given path
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Move model to device and put in eval mode
    device = get_device()
    model.to(device)
    model.eval()

    return model, tokenizer

