from sympy import div
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, BertTokenizerFast
from utils.utils import tokenize, predict
from transformers import BertTokenizerFast
import pickle
import torch.nn.functional as F
import numpy as np
import torch


# def load_bert_model_shap_pruebaaa():
#     from transformers import AutoModelForSequenceClassification
#     return AutoModelForSequenceClassification.from_pretrained("prueba con shap modelo de miguel/Fine_turned/")


def load_bert_model():
    return pickle.load(open("../models/opinion_classification/Bert_Model_Hotel_len_data.pkl", "rb"))


def classify_sentiment_proba(instances):
    from utils.utils import tokenize, predict

    model = load_bert_model()
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "../models/opinion_classification/Fine_turned/")  # shap lime miguel
    # NB
    tokenizer = AutoTokenizer.from_pretrained(
        "models/opinion_classification/Bert_Hotel_max_len/")
    # PY
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "models/opinion_classification/Bert_Hotel_max_len/")
    # Mover el modelo a la GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    probabilities = []
    for review_text in instances:
        if isinstance(review_text, list):
            review_text = ' '.join(review_text)
        encoding_review = tokenize(review_text, tokenizer, 250)
        input_ids = encoding_review['input_ids']
        attention_mask = encoding_review['attention_mask']

        output = model(input_ids, attention_mask)
        proba = F.softmax(output, dim=1)
        Ppos = proba[0][1].item()
        probabilities.append([1-Ppos, Ppos])

    return np.array(probabilities)


def classifySentiment(review_text):
    from utils.utils import tokenize
    import torch
    
    divice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(divice)
    

    model = load_bert_model().to(divice)
    tokenizer = AutoTokenizer.from_pretrained(
        "../models/opinion_classification/Bert_Hotel_max_len/")

    encoding_review = tokenize(review_text, tokenizer, 250)

    input_ids = encoding_review['input_ids'].to(divice)
    attention_mask = encoding_review['attention_mask'].to(divice)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    if prediction.cpu():
        # print(prediction[1],"prediction")
        # print(np.array([1]),"np.array")
        return np.array([1])
    else:
        # print(prediction[0],"prediction")
        # print(np.array([0]),"np.array")
        return np.array([0])


# def classify_sentiment_probaaaaaaaaaa(instances):
#     from utils.utils import tokenize, predict

#     model = load_bert_model()
#     tokenizer = AutoTokenizer.from_pretrained(
#         "models/opinion_classification/Bert_Hotel_max_len/")

#     probabilities = []
#     for review_text in instances:
#         if isinstance(review_text, list):
#             review_text = ' '.join(review_text)
#         elif not isinstance(review_text, str):
#             raise ValueError("Input must be a string or a list of strings.")

#         encoding_review = tokenize(review_text, tokenizer, 250)
#         input_ids = encoding_review['input_ids']
#         attention_mask = encoding_review['attention_mask']

#         with torch.no_grad():
#             output = model(input_ids, attention_mask)
#             proba = F.softmax(output, dim=1)
#             Ppos = proba[0][1].item()
#             probabilities.append([1-Ppos, Ppos])


#     return np.array(probabilities)
