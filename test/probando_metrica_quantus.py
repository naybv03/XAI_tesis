# Importar las bibliotecas necesarias
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.utils import load_dataset, save_explanation
from lime.lime_text import LimeTextExplainer
from models.roberta_model import classify_comment, load_roberta_model_main
from explainability_techniques.lime import LIMEExplainer
import numpy as np

# Cargar el conjunto de datos
# Ajusta la ruta según sea necesario
dataset = load_dataset(
    "models/relevance_classification/dataset/facebook_labeled.csv", sep=",")

# Cargar el modelo RoBERTa preentrenado
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = load_roberta_model_main()

# Configurar el explicador LIME
explainer_roberta = LIMEExplainer(class_names=['not relevant', 'relevant'])

# Seleccionar una instancia del conjunto de datos para explicar
instance_roberta = dataset['Review'].iloc[0]

# Generar la explicación usando LIME
explanation_roberta = explainer_roberta.explain_instance(
    instance_roberta,
    classify_comment  # Asegúrate de que esta función esté definida para clasificar el comentario
)

html_roberta = explanation_roberta.as_html()
save_explanation(html_roberta, 'roberta_explanation.html')
explainer_roberta.save_and_open_html(html_roberta, 'roberta_explanation.html')


# Generar explicaciones para todas las instancias del conjunto de datos
saliency_list = []
intgrad_list = []

# for index in range(len(dataset)):
for index in range(2):

    instance_roberta = dataset['Review'].iloc[index]

    # Generar la explicación usando LIME
    explanation_roberta = explainer_roberta.explain_instance(
        instance_roberta,
        classify_comment  # Asegúrate de que esta función esté definida para clasificar el comentario
    )

    # Extraer las atribuciones de saliencia
    # Obtiene las atribuciones de la clase relevante
    saliency_values = explanation_roberta.as_map()[1]
    saliency_list.append(saliency_values)

# Convertir a arreglos de NumPy y guardar
a_batch_saliency = np.array(saliency_list)
np.save("explanations_lime_roberta.npy", a_batch_saliency)
