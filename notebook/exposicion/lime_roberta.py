# ray Roberta lime
import sys
sys.path.append(
    'D:/Universidad/informatica 4 año/2do Semestre/tesis_XAI')
from utils.utils import load_dataset, save_explanation
from transformers import AutoTokenizer
from models.roberta_model import classify_comment, load_roberta_model_main
from torch import nn
import pickle
import pandas as pd
from lime.lime_text import LimeTextExplainer

from explainability_techniques.lime import LIMEExplainer

dataset = load_dataset(
    "models/relevance_classification/dataset/facebook_labeled.csv", sep=",")

from models.roberta_model import load_roberta_model_notebook


# roberta_model = load_roberta_model_notebook()
roberta_model = load_roberta_model_main()

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
class_names=['not relevant', 'relevant']
explainer_roberta = LimeTextExplainer(class_names=class_names)
instance_index=1

# # Obtener la instancia a explicar desde los argumentos de línea de comandos
# if len(sys.argv) > 1:
#     instance_index = int(sys.argv[1])  # Obtener el índice pasado como argumento
# else:
#     print("No se proporcionó un índice.")
    
# Obtener la instancia del dataset
instance_roberta = dataset['Review'].iloc[instance_index] # type: ignore
print(f"Opinión seleccionada desde lime_roberta: {instance_roberta}")

# instance_roberta = dataset['Review'].iloc[573]
print(instance_index) # type: ignore

explanation_roberta = explainer_roberta.explain_instance(
    instance_roberta, classify_comment)

# Extraer las probabilidades de predicción generadas por LIME
lime_probabilities = explanation_roberta.predict_proba  # Probabilidades generadas por LIME
print(lime_probabilities,"aaaaaaaaaaaaaaaaaaaaaaaaaaaa")
# Preparar los datos para el Excel
data = []
for i, class_name in enumerate(class_names):
    feature_list = []
    weight_list = []
    for feature, weight in explanation_roberta.as_list():
        feature_list.append(feature)
        weight_list.append(weight)
    
    data.append({
        'Instancia': instance_roberta,
        'Clase': class_name,
        'Probabilidad (LIME)': lime_probabilities[i],  # Probabilidad asignada por LIME a esta clase
        'Rasgos': ', '.join(feature_list),  # Convertir lista en string para mejor visualización en Excel
        'Valores de Pertenencia': ', '.join(map(str, weight_list))  # Convertir lista en string
    })

# Crear un DataFrame de pandas
df = pd.DataFrame(data)

# Guardar el DataFrame en un archivo Excel
excel_file = f'lime_explanation_{instance_index}.xlsx'
df.to_excel(excel_file, index=False)

print(f"Explicación guardada en {excel_file}")

# # Preparar los datos para el Excel
# data = []
# for class_name in class_names:
#     feature_list = []
#     weight_list = []
#     for feature, weight in explanation_roberta.as_list():
#         feature_list.append(feature)
#         weight_list.append(weight)
#     data.append({
#         'Instancia': instance_roberta,
#         'Clase': class_name,
#         'Rasgos': feature_list,
#         'Valores de Pertenencia': weight_list
#     })

# # Crear un DataFrame de pandas
# df = pd.DataFrame(data)

# # Guardar el DataFrame en un archivo Excel
# excel_file = f'lime_explanation_{instance_index}.xlsx'
# df.to_excel(excel_file, index=False)

# print(f"Explicación guardada en {excel_file}")

html_roberta = explanation_roberta.as_html()
save_explanation(html_roberta, 'roberta_explanation_lime.html')
explainer_roberta.save_and_open_html(
    html_roberta, 'roberta_explanation_lime.html')