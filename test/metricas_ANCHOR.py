

def calculate_anchor_feature_importance(anchor_explainer, instance):
    """
    Calcula la importancia de las características utilizando el método ANCHOR.

    :param anchor_explainer: Un objeto AnchorText previamente ajustado.
    :param instance: Una instancia (texto) para la cual se desea calcular la importancia.
    :return: Un diccionario con las importancias de las características.
    """
    # Generar una explicación para la instancia 'instance'
    exp = anchor_explainer.explain_instance(
        instance, predict_classify_sentiment, threshold=0.95)

    # Convertir las explicaciones a un diccionario de importancias
    feature_importance = {}
    for feature, importance in exp.as_list():
        feature_importance[feature] = importance

    return feature_importance
