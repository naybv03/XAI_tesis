# from tensorflow.keras.preprocessing.sequence import pad_sequences
import quantus
import scipy as sp
from transformers import AutoModelForSequenceClassification
import pickle
from regex import D
import torch
from lore_explainer.util import record2str
from lore_explainer.lorem import LOREM
from lore_explainer.datamanager import prepare_dataset, prepare_adult_dataset
from explainability_techniques.anchor import AnchorExplainer
# from explainability_techniques.lore import LOREExplainer, save_explanation
from transformers import AutoTokenizer
from utils.utils import load_dataset, save_explanation, predict_relevance_comment
from transformers import AutoTokenizer, BertTokenizerFast
from explainability_techniques.lime import LIMEExplainer
from explainability_techniques.shap import SHAPExplainer
from models.bert_model import load_bert_model, classify_sentiment_proba, classifySentiment
from models.roberta_model import classify_comment, load_roberta_model_main
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import transformers
import spacy
from anchor import anchor_text
import webbrowser
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import shap


# class Rutas:
#     RUTA_NB_lime_bert = "../models/opinion_classification/Bert_Model_Hotel_len_data.pkl"
#     RUTA_PY_lime_bert = "models/opinion_classification/Bert_Model_Hotel_len_data.pkl"
#     RUTA_NB_lime_roberta = '/ruta/a/imagenes'
#     RUTA_PY_lime_roberta = '/ruta/a/imagenes'
#     RUTA_NB_lime_bert_dataset = "../models/opinion_classification/dataset/hotel_review_tain_5000.csv"
#     RUTA_PY_lime_bert_dataset = "models/opinion_classification/dataset/hotel_review_tain_5000.csv"
#     RUTA_NB_lime_roberta_dataset = "../models/relevance_classification/dataset/facebook_labeled.csv"
#     RUTA_PY_lime_roberta_dataset = "models/relevance_classification/dataset/facebook_labeled.csv"

#     RUTA_PY_shap_bert = "Fine_turned/"


# # Carga los datos de prueba preprocesados
# # RoBERTa
# # ray lime
# dataset = load_dataset(Rutas.RUTA_PY_lime_roberta_dataset, sep=",")
# roberta_model = load_roberta_model_main()
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# explainer_roberta = LIMEExplainer(class_names=['not relevant', 'relevant'])
# instance_roberta = dataset['Review'].iloc[0]
# explanation_roberta = explainer_roberta.explain_instance(
#     instance_roberta, classify_comment)
# # explanation_roberta.show_in_notebook()
# # Cálculo de la fidelidad de LIME
# fidelity_score_roberta_lime = calculate_lime_fidelity(
#     explainer=explainer_roberta,
#     model_predict_proba=classify_comment,
#     dataset=dataset,
#     num_samples=min(len(dataset), 2)
# )
# print(fidelity_score_roberta_lime, "fidelity_score_roberta_lime")
# # Cálculo de la importancia de las características según LIME
# lime_feature_importance_roberta = calculate_lime_feature_importance(
#     explainer_roberta, instance_roberta)
# print(lime_feature_importance_roberta, "importancia lime")
# html_roberta = explanation_roberta.as_html()
# save_explanation(html_roberta, 'roberta_explanation.html')
# explainer_roberta.save_and_open_html(html_roberta, 'roberta_explanation.html')

# # -----------------------------------

# # Generar explicaciones para todas las instancias del conjunto de datos
# saliency_list = []
# # intgrad_list = []
# # Inicializar listas para almacenar las opiniones y etiquetas filtradas
# filtered_reviews = []
# filtered_labels = []

# # for index in range(len(dataset)):
# for index in range(3):

#     instance_roberta = dataset['Review'].iloc[index]

#     # Generar la explicación usando LIME
#     explanation_roberta = explainer_roberta.explain_instance(
#         instance_roberta,
#         classify_comment  # Asegúrate de que esta función esté definida para clasificar el comentario
#     )

#     # Extraer las atribuciones de saliencia
#     # Obtiene las atribuciones de la clase relevante
#     saliency_values = explanation_roberta.as_map()[1]

#     if isinstance(saliency_values, list):
#         print(f"Longitud de saliency_values para instancia {
#               index}: {len(saliency_values)}")

#         # Solo agregar si tiene una longitud específica (por ejemplo, 10)
#         if len(saliency_values) == 10:
#             saliency_list.append(saliency_values)
#             # Agregar la reseña filtrada
#             filtered_reviews.append(instance_roberta)
#             # Agregar la etiqueta correspondiente
#             filtered_labels.append(dataset['Relevant'].iloc[index])

# # Convertir a arreglos de NumPy
# a_batch_saliency = np.array(saliency_list)
# x_batch = np.array(filtered_reviews)
# y_batch = np.array(filtered_labels)
# # # Comprobar dimensiones nuevamente
# # print("Dimensiones de a_batch_saliency:", a_batch_saliency.shape)

# # saliency_list.append(saliency_values)

# # # Convertir a arreglos de NumPy y guardar
# # a_batch_saliency = np.array(saliency_list)
# # print("Dimensiones de a_batch_saliency:", a_batch_saliency.shape)

# np.save("explanations_lime_roberta.npy", a_batch_saliency)


# # Dimensiones transformadas: (1407, 10)

# # # Cargar las etiquetas correspondientes (ajusta según tu dataset)
# # y_batch = np.array(dataset['Relevant'])
# # # Las instancias originales
# # x_batch = np.array(dataset['Review'][:len(saliency_list)])
# # Crear una instancia de la métrica Estimación de Fidelidad
# metric_fidelity = quantus.FaithfulnessEstimate()

# print("Dimensiones de x_batch:", x_batch.shape)
# print("Dimensiones de y_batch:", y_batch.shape)
# print("Dimensiones de a_batch_saliency:", a_batch_saliency.shape[2])
# # Transformar a un formato adecuado
# # Seleccionar el primer valor
# a_batch_saliency_reduced = a_batch_saliency[:, :, 0]
# # Verificar dimensiones
# print("Dimensiones transformadas:", a_batch_saliency_reduced.shape)

# # Evaluar la métrica
# scores_fidelity = metric_fidelity(
#     model=roberta_model,
#     x_batch=x_batch,
#     y_batch=y_batch,
#     a_batch=a_batch_saliency_reduced,
#     device='cpu'
# )

# # Imprimir los resultados de la evaluación
# print("Scores de Estimación de Fidelidad:", scores_fidelity)


# ----------------------------------------------------------------------------------------------------------------

# # BERT lime
# # miguel
# dataset = load_dataset(
#     "models/opinion_classification/dataset/hotel_review_tain_5000.csv", sep=";")
# # si no esta esta clase aqui no funciona la carga del modelo ?


# class ModelSentiment(nn.Module):

#     def __init__(self, n_classes):
#         super(ModelSentiment, self).__init__()

#     def forward(self, input_ids, attention_mask):
#         cls_output = self.model(input_ids, attention_mask)
#         drop_output = self.drop(cls_output.pooler_output)
#         output = self.linear(drop_output)
#         return output


# # bert_model = AutoModelForSequenceClassification.from_pretrained(
# #     "Fine_turned/")
# tokenizer = AutoTokenizer.from_pretrained(
#     "models/opinion_classification/Bert_Hotel_max_len/")


# explainer_bert = LIMEExplainer(class_names=['negative', 'positive'])
# instance_bert = dataset['Review'].iloc[0]
# explanation_bert = explainer_bert.explain_instance(
#     instance_bert, classify_sentiment_proba)
# # Cálculo de la fidelidad de LIME
# fidelity_score_bert_lime = calculate_lime_fidelity(
#     explainer=explainer_bert,
#     model_predict_proba=classify_comment,
#     dataset=dataset,
#     num_samples=min(len(dataset), 2)
# )
# print(fidelity_score_bert_lime, "fidelity_score_bert_lime")
# # Cálculo de la importancia de las características según LIME
# lime_feature_importance_roberta = calculate_lime_feature_importance(
#     explainer_bert, instance_bert)
# print(lime_feature_importance_roberta, "importancia lime")
# html_bert = explanation_bert.as_html()
# save_explanation(html_bert, 'bert_explanation.html')
# explainer_bert.save_and_open_html(html_bert, 'bert_explanation.html')


# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# # RoBERTa
# # ray shap
# print("aqui1")
# dataset = load_dataset(
#     "models/relevance_classification/dataset/facebook_labeled.csv", sep=",")

# print("aqui2")

# # roberta_model = load_roberta_model()
# roberta_model = AutoModelForSequenceClassification.from_pretrained(
#     "modelo de ray mejorado(SHAP)/")

# # roberta_model.save_pretrained(
# #     "models/relevance_classification/comment_relevance_detector")
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# print("aqui3")

# # Crear el pipeline de RELEVAMNCIA DE OPINIONES
# classifier = transformers.pipeline("text-classification",
#                                    model=roberta_model, tokenizer=tokenizer)
# review = dataset["Review"].iloc[0]
# print(review)
# prediction = classifier(review)
# print("aqui4", prediction)

# # Crear un objeto SHAP explainer
# explainer = shap.Explainer(classifier)
# print("aqui5")

# # Calcular los valores SHAP para una instancia
# # instance = reviews[0]
# shap_values = explainer([review])
# print("aqui6")

# # Visualizar los valores SHAP
# shap.plots.text(shap_values[0])
# print("aqui7")
# # Esto no abre por aqui pero si en notebook
# # Guardar la explicación en HTML
# shap.save_html("shap_explanation.html", shap_values)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------

# # BERT shap
# # miguel
# shap.initjs()
# dataset = load_dataset(Rutas.RUTA_PY_lime_bert_dataset, sep=";")


# class ModelSentiment(nn.Module):

#     def __init__(self, n_classes):
#         super(ModelSentiment, self).__init__()

#     def forward(self, input_ids, attention_mask):
#         cls_output = self.model(input_ids, attention_mask)
#         drop_output = self.drop(cls_output.pooler_output)
#         output = self.linear(drop_output)
#         return output


# bert_model = AutoModelForSequenceClassification.from_pretrained(
#     Rutas.RUTA_PY_shap_bert)
# tokenizer = AutoTokenizer.from_pretrained(
#     "models/opinion_classification/Bert_Hotel_max_len/")

# # # Crear el pipeline de análisis de sentimientos
# classifier = transformers.pipeline("sentiment-analysis",
#                                    model=bert_model, tokenizer=tokenizer, return_all_scores=True)

# review = dataset["Review"].iloc[:5]

# # define a prediction function


# def f(x):
#     # print(x)
#     tv = torch.tensor([tokenizer.encode(
#         v, padding="max_length", max_length=500, truncation=True) for v in x])
#     outputs = bert_model(tv)[0].detach().numpy()
#     scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
#     val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
#     # print(scores[:, 0], " Proba")
#     return val


# # build an explainer using a token masker
# explainer = shap.Explainer(f, tokenizer)

# # explain the model's predictions on IMDB reviews
# shap_values = explainer(review[:2], fixed_context=1, batch_size=10)

# shap.text_plot(shap_values)


# # Calcular la fidelidad de SHAP

# # Paso 1: Definir la función para modificar instancias


# def modify_instance_based_on_shap(instance, shap_values_instance, num_features_to_modify):
#     # Dividir la cadena en palabras para poder modificarla
#     instance_words = instance.split()  # Divide la cadena en palabras
#     # Hacer una copia de la lista de palabras
#     instance_perturbed = instance_words.copy()

#     # Solo vamos a modificar las 'num_features_to_modify' características más importantes
#     for i in np.argsort(np.abs(shap_values_instance))[-num_features_to_modify:]:
#         if i < len(instance_perturbed):  # Asegúrate de no exceder el rango
#             # Si es numérico (esto no aplicará aquí)
#             if isinstance(instance_perturbed[i], (int, float)):
#                 # Establecer el valor en el promedio de esa característica (ajusta según tu contexto)
#                 # Convertir a string si es necesario
#                 instance_perturbed[i] = str(np.mean(dataset.iloc[:, i]))
#             # Si es categórico (en este caso, una palabra)
#             elif isinstance(instance_perturbed[i], str):
#                 # Reemplazar con un valor que represente "no presente"
#                 # O cualquier otro valor que tenga sentido
#                 instance_perturbed[i] = "neutral"

#     # Volver a unir las palabras en una cadena
#     return ' '.join(instance_perturbed)  # Vuelve a unir las palabras


# # Paso 2: Calcular la fidelidad
# accuracies = []

# # Cambia el rango según cuántas instancias quieras evaluar
# for i in range(min(len(review[:2]), len(shap_values.values))):
#     instance = review.iloc[i]

#     # Obtener los valores SHAP para esta instancia
#     # print(shap_values[i].values, "   shap_values[i].values")
#     shap_value_instance = shap_values[i].values

#     # Modificar la instancia basada en los valores SHAP
#     instance_perturbed = modify_instance_based_on_shap(
#         instance, shap_value_instance, num_features_to_modify=5)

#     # Obtener predicciones originales y perturbadas
#     original_pred = np.argmax(f([instance]))  # Predicción original
#     perturbed_pred = np.argmax(
#         f([instance_perturbed]))  # Predicción perturbada

#     # Comparar y almacenar el resultado
#     accuracies.append(original_pred == perturbed_pred)


# # Paso 3: Calcular la fidelidad promedio
# print(accuracies)
# fidelity_shap = np.mean(accuracies)
# print(f"Fidelidad de las explicaciones de SHAP: {fidelity_shap}")


# # Crear un objeto SHAP explainer
# shap_explainer = SHAPExplainer(classifier, tokenizer)

# # Explicar una instancia
# shap_values = shap_explainer.explain_instance(review[:1])

# shap.plots.text(shap_values)
# plt.show()
# # Guardar y abrir la explicación en HTML
# shap_explainer.save_and_open_html(shap_values, "shap_explanation.html")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# BERT anchor
# miguel
dataset = load_dataset(
    "models/opinion_classification/dataset/hotel_review_tain_5000.csv", sep=";")
# si no esta esta clase aqui no funciona la carga del modelo, cosas de miguel


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

# Crear el explicador de Anchors
anchor_explainer = AnchorExplainer(class_names=['negative', 'positive'])


# Seleccionar una instancia para explicar
instance = dataset['Review'].iloc[0]
print(instance)
print(classifySentiment(instance), " prediccion")


def predict_classify_sentiment(instance):
    if isinstance(instance, list):
        print("entraaaaaa")
        instance = instance[0]  # Extraer el primer elemento si es una lista
    return classifySentiment(instance)


# Explicar la instancia
exp = anchor_explainer.explain_instance(
    instance, predict_classify_sentiment, threshold=0.95)

# Mostrar resultados
anchor_explainer.print_results(exp)

# Guardar y abrir el HTML
anchor_explainer.save_and_open_html(exp, "explanation.html")

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# # Carga los datos de prueba preprocesados
# # ray anchors
# dataset = load_dataset(
#     "models/relevance_classification/dataset/facebook_labeled.csv", sep=",")

# # RoBERTa
# roberta_model = torch.load(
#     open("models/relevance_classification/comment_relevance_detector (facebook).pth", 'rb'))
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# # Crear el explicador de Anchors
# anchor_explainer = AnchorExplainer(class_names=['not relevant', 'relevant'])

# # Seleccionar una instancia para explicar
# instance = dataset['Review'].iloc[1790]
# print(instance)
# print(predict_relevance_comment(instance, roberta_model, tokenizer), " prediccion")


# def predict_relevance(instance):
#     if isinstance(instance, list):
#         print("entraaaaaa")
#         instance = instance[0]  # Extraer el primer elemento si es una lista
#     return predict_relevance_comment(instance, roberta_model, tokenizer)


# # Explicar la instancia
# exp = anchor_explainer.explain_instance(
#     instance, predict_relevance, threshold=0.95)

# # Mostrar resultados
# anchor_explainer.print_results(exp)

# # Guardar y abrir el HTML
# anchor_explainer.save_and_open_html(exp, "explanation.html")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Carga los datos de prueba preprocesados
# # ray lore
# dataset = load_dataset(
#     "models/relevance_classification/dataset/facebook_labeled.csv", sep=",")
# # dataset, class_name = prepare_adult_dataset(
# #     "models/relevance_classification/dataset/facebook_labeled.csv", sep=",")
# print(dataset.shape, "1")

# # RoBERTa
# roberta_model = load_roberta_model()
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# class_name = 'Relevant'  # Nombre de la clase objetivo
# # print(dataset.columns)
# # Preparar datos para LORE
# # X_train = np.array(dataset['Review'].tolist())
# # y_train = np.array(dataset['Relevant'].tolist())
# dataset, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(
#     dataset, class_name)
# print(dataset.shape)
# # print(dataset.columns)


# test_size = 0.20
# random_state = 0
# # print(class_name)
# # print(dataset['Relevant'].value_counts())
# # print(dataset.isnull().sum())
# # Transformar los datos de texto a numéricos


# # print("algooo", dataset[class_name].values)

# X_train, X_test, Y_train, Y_test = train_test_split(dataset[feature_names].values, dataset[class_name].values,
#                                                     test_size=test_size,
#                                                     random_state=random_state,
#                                                     stratify=dataset[class_name].values)

# _, K, _, _ = train_test_split(rdf[real_feature_names].values, rdf[class_name].values,
#                               test_size=test_size,
#                               random_state=random_state,
#                               stratify=dataset[class_name].values)
# print("aquiiii1")
# # print(X_train)
# # print(X_test)
# # print(Y_train)
# # print(Y_test)
# # Y_pred = bb_predict(X_test)

# # print('Accuracy %.3f' % accuracy_score(Y_test, Y_pred))
# # print('F1-measure %.3f' % f1_score(Y_test, Y_pred))
# vectorizer = TfidfVectorizer()

# vectorizer.fit(dataset['Review'])
# i2e = 3
# x_text = dataset['Review'][i2e]
# print("aquiiii2", x_text)

# x_tfidf = vectorizer.transform([x_text])
# # print("Reseña transformada a TF-IDF:")
# # print(x_tfidf)
# # print("aque tenrminaaaaaaaaaaaa")
# # Convertir la matriz dispersa a un formato denso
# x_dense = x_tfidf.toarray()
# # Asegurarte de que x_dense esté en el formato correcto
# print(type(x_dense), "este es el tipo")
# print(x_dense, "este es el valor")
# # Verificar el formato de x_dense[0]
# print(type(x_dense[0]))
# print(x_dense[0])
# # Si X_test es un arreglo de NumPy, conviértelo a un DataFrame de pandas para una mejor visualización
# # Asegúrate de que feature_names contenga los nombres de las características
# print("Etiquetas de clase [0, 1]")

# df_X_test = pd.DataFrame(X_test, columns=feature_names)
# print(df_X_test.head())  # Muestra las primeras 5 filas
# print('Instancia correpondiente a explicar x = %s' %
#       record2str(x_dense[0], feature_names, numeric_columns))
# print('x?text', x_text)
# print(x_dense.shape)

# bb_outcome = bb_predict(x_dense)
# bb_outcome_str = class_values[bb_outcome[0]]

# print(
#     'Clase correspondiente a la prediccion de black box => bb(x) = { %s }' % bb_outcome_str)
# print('')

# print(bb_predict_proba(x_dense))
# explainer_lore = LOREExplainer(K, bb_predict, feature_names, class_name, class_values, numeric_columns, features_map, neigh_type='geneticp',
#                                categorical_use_prob=True, continuous_fun_estimation=False, size=1000, ocr=0.1, ngen=1,  bb_predict_proba=bb_predict_proba, verbose=True)

# # print(type(x_text))
# # print(x_text)

# explanation_lore = explainer_lore.explain_instance(x_dense, samples=1000)
# print(explanation_lore)

# # html_lore = explanation_lore.as_html()
# # save_explanation(html_lore, 'lore_explanation.html')
# # explainer_lore.save_and_open_html(html_lore, 'lore_explanation.html')

# ----------------------------------------------------------------------------------------------------------------

# BERT lore
# miguel
# dataset = load_dataset(
#     "models/opinion_classification/dataset/hotel_review_tain_5000.csv", sep=";")
# # si no esta esta clase aqui no funciona la carga del modelo ?


# class ModelSentiment(nn.Module):

#     def __init__(self, n_classes):
#         super(ModelSentiment, self).__init__()

#     def forward(self, input_ids, attention_mask):
#         cls_output = self.model(input_ids, attention_mask)
#         drop_output = self.drop(cls_output.pooler_output)
#         output = self.linear(drop_output)
#         return output


# bert_model = load_bert_model()
# tokenizer = AutoTokenizer.from_pretrained(
#     "models/opinion_classification/Bert_Hotel_max_len/")


# explainer_bert = Explainer(class_names=['negative', 'positive'])
# instance_bert = dataset['Review'].iloc[0]
# explanation_bert = explainer_bert.explain_instance(
#     instance_bert, classify_sentiment)
# html_bert = explanation_bert.as_html()
# save_explanation(html_bert, 'bert_explanation.html')
# explainer_bert.save_and_open_html(html_bert, 'bert_explanation.html')
