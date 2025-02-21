import spacy
from anchor.anchor_text import AnchorText
from utils.utils import save_explanation
import webbrowser


class AnchorExplainer:
    def __init__(self, class_names):
        self.nlp = spacy.load('en_core_web_sm')
        self.explainer = AnchorText(
            self.nlp, class_names, use_unk_distribution=True)

    def explain_instance(self, instance, predict_function, threshold=0.95):
        return self.explainer.explain_instance(
            instance,
            predict_function,
            threshold=threshold
        )

    def save_and_open_html(self, html_content, file_name):
        save_explanation(html_content, file_name)
        webbrowser.open_new_tab(file_name)

    def print_results(self, exp):
        print('Prediction: %s' % exp.class_names[exp.predicted_class])
        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())

        self.print_examples(exp)

    def print_examples(self, exp):
        pred_class = exp.predicted_class
        alternative = exp.class_names[1 - pred_class]

        print('Examples where anchor applies and model predicts %s:' %
              exp.class_names[pred_class])
        print('\n'.join([x[0]
              for x in exp.examples(only_same_prediction=True)]))

        print('Examples where anchor applies and model predicts %s:' % alternative)
        print('\n'.join([x[0] for x in exp.examples(
            partial_index=0, only_different_prediction=True)]))
