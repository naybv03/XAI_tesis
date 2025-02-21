import sys
sys.path.append(
    'D:/Universidad/informatica 4 año/2do Semestre/tesis_XAI')
import numpy as np
from explainability_techniques.lime import LIMEExplainer
from transformers import AutoTokenizer
from models.bert_model import load_bert_model, classify_sentiment_proba
from torch import nn
import pickle
from utils.utils import load_dataset, save_explanation


dataset = load_dataset(
    "models/opinion_classification/dataset/hotel_review_tain_5000.csv", sep=";")


class ModelSentiment(nn.Module):

    def __init__(self, n_classes):
        super(ModelSentiment, self).__init__()

    def forward(self, input_ids, attention_mask):
        cls_output = self.model(input_ids, attention_mask)
        drop_output = self.drop(cls_output.pooler_output)
        output = self.linear(drop_output)
        return output


bert_model = pickle.load(
    open("models/opinion_classification/Bert_Model_Hotel_len_data.pkl", "rb"))


tokenizer = AutoTokenizer.from_pretrained(
    "models/opinion_classification/Bert_Hotel_max_len/")

explainer_bert = LIMEExplainer(class_names=['negative', 'positive'])

# Obtener la instancia a explicar desde los argumentos de línea de comandos
if len(sys.argv) > 1:
    instance_index = int(sys.argv[1])  # Obtener el índice pasado como argumento
else:
    print("No se proporcionó un índice.")
    
# Obtener la instancia del dataset
instance_bert = dataset['Review'].iloc[0] # type: ignore
print(f"Opinión seleccionada desde lime_bert: {instance_bert}")
# print(instance_bert)

explanation_bert = explainer_bert.explain_instance(
    instance_bert, classify_sentiment_proba)


html_bert = explanation_bert.as_html()
save_explanation(html_bert, 'bert_explanation_lime.html')
explainer_bert.save_and_open_html(
    html_bert, 'bert_explanation_lime.html')
