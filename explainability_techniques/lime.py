from lime.lime_text import LimeTextExplainer
import webbrowser
from utils.utils import save_explanation
class LIMEExplainer:
    def __init__(self, class_names):
        self.explainer = LimeTextExplainer(class_names=class_names)

    def explain_instance(self, instance, predict_function, labels=(1,), num_features=10, num_samples=100):
        return self.explainer.explain_instance(
            instance,
            predict_function,
            labels=labels,
            num_features=num_features,
            num_samples=num_samples
        )

    def get_predict_proba(self, explanation):
        # Obtener las probabilidades de predicción desde la explicación
        probabilities = explanation.predict_proba  # Obtener las probabilidades
        return probabilities  # Devolver las probabilidades
    
    def save_and_open_html(self, html_content, file_name):
        save_explanation(html_content, file_name)
        webbrowser.open_new_tab(file_name)
