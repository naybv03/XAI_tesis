# ray RoBerta shap
import sys
sys.path.append(
    'D:/Universidad/informatica 4 año/Semestre 1/Precticas 2/PP2_XAI')
import scipy as sp
import torch
from explainability_techniques.shap import SHAPExplainer
import shap
import numpy as np
import torch.nn as nn
import transformers
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import webbrowser
import pandas as pd
from transformers import AutoModelForSequenceClassification


shap.initjs()

dataset = pd.read_csv(
    "../models/relevance_classification/dataset/facebook_labeled.csv", sep=",")

roberta_model = AutoModelForSequenceClassification.from_pretrained(
    "../models/relevance_classification/modelo de ray mejorado(SHAP)/")


tokenizer = AutoTokenizer.from_pretrained("roberta-base")

review = dataset["Review"]
# 1024, 1075, 1159, 560, 517, 1047,596
posiciones = [1024]


reseñas_seleccionadas = review.iloc[posiciones]


# print(dataset["Review"][:20])
reseñas_lista = reseñas_seleccionadas.tolist()
print(reseñas_lista)

classifier = transformers.pipeline(
    "text-classification", model=roberta_model, tokenizer=tokenizer, return_all_scores=True)

classifier(reseñas_lista)

shap_explainer = SHAPExplainer(classifier)

# Explicar una instancia
print(reseñas_lista)

shap_values = shap_explainer.explain_instance(reseñas_lista)

print(shap_values.shape)


shap.plots.text(shap_values)
# 596,1024