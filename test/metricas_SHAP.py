

from models.roberta_model import classify_comment
import numpy as np
from explainability_techniques.lime import LIMEExplainer

# # Modificación de instancias basadas en LIME


# def modify_instance_based_on_lime(instance, lime_exp, num_perturbations=10):
#     perturbed_instances = []
#     for _ in range(num_perturbations):
#         perturbed_instance = instance.copy()
#         perturbed_instances.append(perturbed_instance)
#     return perturbed_instances


# def modify_instance_based_on_lime_new(instance, lime_exp, num_perturbations=10):
#     perturbed_instances = []
#     instance_array = np.array(instance.split())

#     # Crear un diccionario para mapear palabras a sus índices
#     word_to_index = {word: idx for idx, word in enumerate(instance_array)}
#     # print(word_to_index, "indice y palabra")
#     for _ in range(num_perturbations):
#         perturbed_instance = instance_array.copy()

#         for feature, importance in lime_exp:
#             if importance > 0 and feature in word_to_index:
#                 feature_index = word_to_index[feature]
#                 perturbed_instance[feature_index] = "PERTURBADO"

#         perturbed_instances.append(" ".join(perturbed_instance))
#     return perturbed_instances


# # Cálculo de la fidelidad de LIME


# def calculate_lime_fidelity(explainer, model_predict_proba, dataset, num_samples=100):
#     print("entro")
#     fidelity_scores = []

#     for i in range(num_samples):
#         test_instance = dataset['Review'].iloc[i]
#         print(test_instance, "-->Se selecciona una instancia de prueba test_instance del conjunto de datos.")
#         predicted_label = np.argmax(model_predict_proba([test_instance])[0])
#         print(predicted_label, "  -->Se obtiene la etiqueta predicha predicted_label utilizando la función de predicción del modelo.")

#         exp = explainer.explain_instance(
#             instance=test_instance,
#             predict_function=model_predict_proba,
#             labels=[predicted_label],
#             num_features=10  # Puedes ajustar este valor según sea necesario
#         )
#         lime_exp = exp.as_list(label=predicted_label)
#         # print(lime_exp, "tuplaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
#         modified_instances = modify_instance_based_on_lime_new(
#             test_instance, lime_exp)

#         if modified_instances is not None:
#             # original_prediction = model_predict_proba([test_instance])[0]
#             perturbed_predictions = np.array([model_predict_proba(
#                 [instance])[0] for instance in modified_instances])

#             fidelity = np.mean(
#                 np.argmax(perturbed_predictions, axis=1) == predicted_label)
#             fidelity_scores.append(fidelity)

#     return np.mean(fidelity_scores)

# Cálculo de la importancia de las características según SHAP


def calculate_shap_feature_importance(shap_explainer, x_instance):
    shap_values = shap_explainer.shap_values(x_instance)
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    return feature_importance
