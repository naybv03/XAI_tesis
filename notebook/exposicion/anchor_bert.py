# miguel BERT anchor
import sys
sys.path.append(
    'D:/Universidad/informatica 4 año/2do Semestre/tesis_XAI')
from transformers import AutoModelForSequenceClassification
from utils.utils import load_dataset, save_explanation
from explainability_techniques.anchor import AnchorExplainer
from transformers import AutoTokenizer
from models.bert_model import load_bert_model, classifySentiment
from torch import nn
import pickle


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
    

bert_model = AutoModelForSequenceClassification.from_pretrained(
    "models/opinion_classification/Fine_turned/").to('cuda') 
tokenizer = AutoTokenizer.from_pretrained(
    "models/opinion_classification/Bert_Hotel_max_len/")


# Crear el explicador de Anchors
anchor_explainer = AnchorExplainer(class_names=['negative', 'positive'])

# Obtener la instancia a explicar desde los argumentos de línea de comandos
if len(sys.argv) > 1:
    instance_index = int(sys.argv[1])  # Obtener el índice pasado como argumento
else:
    print("No se proporcionó un índice.")
    
# Obtener la instancia del dataset
instance_anchor = dataset['Review'].iloc[instance_index] # type: ignore
print(f"Opinión seleccionada desde anchor_bert: {instance_anchor}")
# Seleccionar una instancia para explicar
# instance = dataset['Review'].iloc[26]
print(instance_anchor)

def predict_classify_sentiment(instance):
    # if isinstance(instance, list):
    #     print("entraaaaaa")
    #     instance = instance[0]  # Extraer el primer elemento si es una lista
    return classifySentiment(instance)

# Explicar l a instancia
exp = anchor_explainer.explain_instance(
    instance_anchor, predict_classify_sentiment, threshold=0.95)

# Guardar y abrir el HTML
html_bert = exp.as_html()
save_explanation(html_bert, 'bert_explanation_anchor.html')

anchor_explainer.save_and_open_html(
    exp.as_html(), "bert_explanation_anchor.html")