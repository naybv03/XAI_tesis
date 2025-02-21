# miguel Bert shap
import sys
sys.path.append(
    'D:/Universidad/informatica 4 año/Semestre 1/Precticas 2/PP2_XAI')
from models.bert_model import classify_sentiment_proba
import pandas as pd
import webbrowser
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import transformers
import torch.nn as nn
import numpy as np
import torch
import scipy as sp
import shap
from explainability_techniques.shap import SHAPExplainer
from transformers import AutoModelForSequenceClassification

shap.initjs()

dataset = pd.read_csv(
    "models/opinion_classification/dataset/hotel_review_tain_5000.csv", sep=";")

bert_model = AutoModelForSequenceClassification.from_pretrained(
    "models/opinion_classification/Fine_turned/")


tokenizer = AutoTokenizer.from_pretrained(
    "models/opinion_classification/Bert_Hotel_max_len/")

review = dataset["Review"]
# 5,6,7,10,1510, 3319, 3320, 3322
posiciones = [6]
reseñas_seleccionadas = review.iloc[posiciones]
reseñas_lista = reseñas_seleccionadas.tolist()

print(reseñas_lista)

classifier = transformers.pipeline(
    "sentiment-analysis", model=bert_model, tokenizer=tokenizer, return_all_scores=True)

classifier(reseñas_lista)

shap_explainer = SHAPExplainer(classifier)

shap_values = shap_explainer.explain_instance(reseñas_lista)

print(len(reseñas_lista))

print(shap_values.shape)
# 6,7,10
shap.plots.text(shap_values)
