import shap
import webbrowser
from utils.utils import save_explanation


class SHAPExplainer:
    def __init__(self, classify_sentiment_proba, tokenizer):
        self.explainer = shap.Explainer(classify_sentiment_proba, tokenizer)

    def explain_instance(self, instance):
        shap_values = self.explainer(instance)
        return shap_values

    def save_and_open_html(self, shap_values, file_name):
        shap_html = shap.plots.text(shap_values)
        if shap_html is None:
            print("Error: shap_html es None")
        else:
            print("shap_html generado correctamente")
            save_explanation(shap_html, file_name)
            webbrowser.open_new_tab(file_name)
