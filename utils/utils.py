import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from models.roberta_model import classify_comment


def tokenize(text, tokenizer, max_length):
    # Asegúrate de que text sea un string
    if isinstance(text, list):
        text = text[0]  # Tomar el primer elemento si es una lista
    elif not isinstance(text, str):
        raise ValueError("No es una lista")
    try:
        return tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
    except Exception as e:
        print(f"Error al tokenizar: {e}")
        raise


def predict(model, input_ids, attention_mask):
    _, output = model(input_ids, attention_mask)
    logits = output[0]
    if isinstance(logits, int):
        logits = torch.tensor([logits], dtype=torch.float32)

    probs = logits.flatten().item()
    return [1 - probs, probs]


# def predict_relevance_comment(instance, model, tokenizer):
#     encoding = tokenize(instance, tokenizer, 512)

#     _, prediction = model(
#         encoding["input_ids"], encoding["attention_mask"])
#     prediction = prediction.flatten().item()

#     return [1] if prediction >= 0.5 else [0]

def predict_relevance_comment(instance, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Asegurar que el modelo está en el dispositivo correcto
    model.eval()  # Poner el modelo en modo de evaluación
    print(device,"entro 1")
    encoding = tokenize(instance, tokenizer, 512)

    with torch.no_grad():
        print("entro 2")
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device))

    print("entro 3")
    # Verificar si outputs es una tupla y obtener logits
    logits = outputs[1] if isinstance(outputs, tuple) else outputs.logits
    

    print("entro 4")
    # Asegurarse de que logits sea un tensor
    if isinstance(logits, torch.Tensor):
        print("Shape of logits:", logits.shape)
        print("entro",logits)

        print("entro 5.1",logits[0][0].item())
        prediction = logits[0][0].item()  # Toma el primer elemento
        predicted_class = 1 if prediction >= 0.5 else 0
        return np.array([predicted_class])

    else:
        raise ValueError("Logits no es un tensor")


def load_dataset(file_path, sep):
    return pd.read_csv(file_path, sep=sep)


def save_explanation(html_content, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(html_content)
