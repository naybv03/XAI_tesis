# from utils.utils import tokenize, predict
from transformers import AutoTokenizer
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

def load_roberta_model_notebook():
    return torch.load(
        open("../models/relevance_classification/comment_relevance_detector (facebook).pth", 'rb'))


def load_roberta_model_main():
    return AutoModelForSequenceClassification.from_pretrained("D:/Universidad/informatica 4 año/2do Semestre/tesis_XAI/models/relevance_classification/modelo de ray mejorado_SHAP/", weights_only=False)


def classify_comment(instances):
    from utils.utils import tokenize, predict

    # model = load_roberta_model_main()
    model = load_roberta_model_notebook()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    probabilities = []

    for review_text in instances:
        if isinstance(review_text, list):
            review_text = ' '.join(review_text)
        encoding_review = tokenize(review_text, tokenizer, 512)
        input_ids = encoding_review['input_ids']
        attention_mask = encoding_review['attention_mask']

        probs = predict(model, input_ids, attention_mask)
        probabilities.append(probs)

    return np.array(probabilities)
