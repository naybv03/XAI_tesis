# ray Roberta anchor
import sys
sys.path.append(
    'D:/Universidad/informatica 4 año/2do Semestre/tesis_XAI')
from explainability_techniques.anchor import AnchorExplainer
from transformers import AutoTokenizer
from models.roberta_model import load_roberta_model_main, load_roberta_model_notebook
import torch 
import pickle
import pandas as pd
from utils.utils import load_dataset, save_explanation, predict_relevance_comment
# Verificar si hay una GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
devNumber=torch.cuda.current_device()
print(f"Current divice number is: {devNumber}")
devName=torch.cuda.get_device_name(devNumber)
print(f"GPU name is: {devName}")

dataset = pd.read_csv(
    "models/relevance_classification/dataset/facebook_labeled.csv", sep=",")

roberta_model = load_roberta_model_main()

tokenizer = AutoTokenizer.from_pretrained("roberta-base")


# Crear el explicador de Anchors
anchor_explainer = AnchorExplainer(class_names=['not relevant', 'relevant'])

# Seleccionar una instancia para explicar
print(len(dataset['Review']))

# # Obtener la instancia a explicar desde los argumentos de línea de comandos
# if len(sys.argv) > 1:
#     instance_index = int(sys.argv[1])  # Obtener el índice pasado como argumento
# else:
#     print("No se proporcionó un índice.")
instance_index=1  
# Obtener la instancia del dataset
instance_anchor = dataset['Review'].iloc[instance_index] # type: ignore
print(f"Opinión seleccionada desde anchor_bert: {instance_anchor}")
# instance = dataset['Review'].iloc[596]
print(instance_anchor)



def predict_relevance(instance):
    # if isinstance(instance, list):
    #     print("entraaaaaa")
    #     instance = instance[0]  # Extraer el primer elemento si es una lista
    return predict_relevance_comment(instance, roberta_model, tokenizer)


# Explicar la instancia
exp = anchor_explainer.explain_instance(
    instance_anchor, predict_relevance, threshold=0.95)


# print(dir(exp))
# Guardar y abrir el HTML
html_roberta = exp.as_html()
save_explanation(html_roberta, 'roberta_explanation_opinion596_anchor.html')
anchor_explainer.save_and_open_html(
    exp.as_html(), "roberta_explanation_opinion596_anchor.html")