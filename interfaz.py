import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import joblib  # Para cargar modelos
import lime
import shap
import pandas as pd
from models.roberta_model import classify_comment, load_roberta_model_main
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from utils.utils import load_dataset
import subprocess
import os
import torch

# Cargar los modelos preentrenados
# roberta_model = load_roberta_model_main()
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# bert_model = AutoModelForSequenceClassification.from_pretrained(
#     "models/opinion_classification/Fine_turned/")
# tokenizer = AutoTokenizer.from_pretrained(
    # "models/opinion_classification/Bert_Hotel_max_len/")
# Variables globales
dataset = None
selected_opinion = None


def load_datasett():
    global dataset
    if model_var.get() == "Clasificación":
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")])
        # print(file_path)
        if file_path:
            dataset = load_dataset(file_path, sep=";")
            opinion_list.delete(0, tk.END)  # Limpiar el Listbox

            # Cargar opiniones con índices en el Listbox
            for index, opinion in enumerate(dataset['Review']):
                # Formato "índice: opinión"
                opinion_list.insert(tk.END, f"{index}:   {opinion}")

            messagebox.showinfo(
                "Carga de Dataset", "Dataset ha sido cargado exitosamente.")
    elif model_var.get() == "Relevancia":
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")])
        if file_path:
            dataset = load_dataset(file_path, sep=",")
            opinion_list.delete(0, tk.END)

            # Cargar opiniones con índices en el Listbox
            for index, opinion in enumerate(dataset['Review']):
                # Formato "índice: opinión"
                opinion_list.insert(tk.END, f"{index}:   {opinion}")

            messagebox.showinfo("Carga de Dataset",
                                "Dataset ha sido cargado exitosamente.")


def select_opinion(event):
    global selected_opinion
    selection = opinion_list.curselection()
    if selection:
        selected_opinion = opinion_list.get(selection[0]).split(
            ":   ", 1)[1]  # Obtener solo la opinión después del índice
        print("Opinión seleccionada:", selected_opinion)


def explain_model():
    if not selected_opinion:
        messagebox.showwarning(
            "Selección de Opinión", "Por favor selecciona una opinión para explicar.")
        return

    selected_model = model_var.get()
    explanation_method = method_var.get()

    # if selected_model == "Clasificación":
    #     model = bert_model
    # else:
    #     model = roberta_model

    try:
        # Obtener el índice de la opinión seleccionada
        selection = opinion_list.curselection()
        if selection:
            opinion_index = selection[0]  # Guardar solo el índice del Listbox
            print(opinion_index)

            # # Deshabilitar el botón y mostrar estado
            # explain_button.config(state=tk.DISABLED)
            # status_label.config(text="Generando explicación...")
            # ver bien como hacer esto 
            
        if explanation_method == "LIME":
            if selected_model == "Clasificación":
                print("Explicación a Clasificación con LIME generada.")

                subprocess.run([
                    "python", 
                    "D:/Universidad/informatica 4 año/Semestre 1/Precticas 2/PP2_XAI/notebook/exposicion/lime_bert.py", 
                    str(opinion_index)  # Pasar el índice seleccionado como argumento # type: ignore
                ], check=True)
            else:
                print("Explicación a Relevancia con LIME generada.")
                subprocess.run([
                    "python", 
                    "D:/Universidad/informatica 4 año/Semestre 1/Precticas 2/PP2_XAI/notebook/exposicion/lime_roberta.py", 
                    str(opinion_index)  # Pasar el índice seleccionado como argumento # type: ignore
                ], check=True)
                

        # elif explanation_method == "SHAP":
        #     if selected_model == "Clasificación":
        #         print("Explicación a Clasificación con SHAP generada.")
        #         subprocess.run([
        #             "python", 
        #             "D:/Universidad/informatica 4 año/Semestre 1/Precticas 2/PP2_XAI/notebook/exposicion/shap_bert.py", 
        #             str(opinion_index)  # Pasar el índice seleccionado como argumento # type: ignore
        #         ], check=True)

        #     else:
        #         print("Explicación a Relevancia con SHAP generada.")
        #         subprocess.run([
        #             "python", 
        #             "D:/Universidad/informatica 4 año/Semestre 1/Precticas 2/PP2_XAI/notebook/exposicion/shap_roberta.py", 
        #             str(opinion_index)  # Pasar el índice seleccionado como argumento # type: ignore
        #         ], check=True)
        elif explanation_method == "Anchor":
            if selected_model == "Clasificación":
                print("Explicación a Clasificación con Anchor generada.")
                subprocess.run([
                    "python", 
                    "D:/Universidad/informatica 4 año/Semestre 1/Precticas 2/PP2_XAI/notebook/exposicion/anchor_bert.py", 
                    str(opinion_index)  # Pasar el índice seleccionado como argumento # type: ignore
                ], check=True)
            else:
                print("Explicación a Relevancia con Anchor generada.")
                subprocess.run([
                    "python", 
                    "D:/Universidad/informatica 4 año/Semestre 1/Precticas 2/PP2_XAI/notebook/exposicion/anchor_roberta.py", 
                    str(opinion_index)  # Pasar el índice seleccionado como argumento # type: ignore
                ], check=True)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Función para limpiar el Listbox
def clear_opinion_list():
    opinion_list.delete(0, tk.END)  # Limpiar todos los elementos del Listbox
    global selected_opinion
    selected_opinion = None  # Reiniciar la selección de opinión


# Crear la ventana principal
root = tk.Tk()
root.title("Explicabilidad de Modelos")

# Obtener las dimensiones de la pantalla
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Configurar el tamaño de la ventana
root.geometry(f"{screen_width}x{screen_height}+0+0")  # Ocupa toda la pantalla
# Deshabilitar el botón de maximizar
root.resizable(False, False)

# Estilo de ttk
style = ttk.Style()
style.configure("TButton", padding=6, relief="flat",
                background="#007BFF", foreground="black")
style.configure("TLabel", font=("Arial", 12), foreground="black")
style.configure("TOptionMenu", font=("Arial", 12))

# Frame principal para organizar los widgets
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Espaciado superior para centrar verticalmente los elementos
padding_top = 20

# Selección del modelo
model_var = tk.StringVar(value="Clasificación")
model_label = ttk.Label(main_frame, text="Selecciona el modelo:")
model_label.grid(row=0, column=0, padx=10, pady=(padding_top, 5), sticky='w')
model_menu = ttk.OptionMenu(
    main_frame, model_var, "Clasificación", "Clasificación", "Relevancia")
model_menu.grid(row=0, column=1, padx=10, pady=(padding_top, 5), sticky='w')

# Cargar dataset
load_button = ttk.Button(
    main_frame, text="Cargar Dataset", command=load_datasett)
load_button.grid(row=0, column=3, columnspan=2, pady=(10, 15))


def on_model_change(*args):
    clear_opinion_list()  # Limpiar el Listbox cuando cambia el modelo

# Rastrear cambios en model_var
model_var.trace_add("write", on_model_change)

# Listbox para mostrar opiniones junto con sus índices
opinion_list_label = ttk.Label(main_frame, text="Opiniones:")
opinion_list_label.grid(row=2, column=0, padx=10, pady=(5, 5), sticky='e')
opinion_list = tk.Listbox(main_frame, width=210, height=30)
opinion_list.grid(row=2, column=4, padx=10, pady=(5, 5))
opinion_list.bind('<<ListboxSelect>>', select_opinion)

# Selección del método explicativo
method_var = tk.StringVar(value="LIME")
method_label = ttk.Label(main_frame, text="Selecciona el método explicativo:")
method_label.grid(row=3, column=0, padx=10, pady=(5, 5), sticky='w')
method_menu = ttk.OptionMenu(
    main_frame, method_var, "LIME", "LIME", "SHAP", "Anchor")
method_menu.grid(row=3, column=1, padx=10, pady=(5, 5))

# Botón para generar la explicación
explain_button = ttk.Button(
    main_frame, text="Generar Explicación", command=explain_model)
explain_button.grid(row=3, column=3, columnspan=2, pady=(15))


# Crear un Label para mostrar el estado de carga
status_label = ttk.Label(main_frame, text="", font=("Arial", 12))
status_label.grid(row=5, column=3, columnspan=2, pady=(15))





root.mainloop()
