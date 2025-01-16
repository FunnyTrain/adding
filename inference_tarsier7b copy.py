import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForImageText, AutoTokenizer, AutoProcessor

# Cargar el modelo y tokenizador con el tipo adecuado
model_name = "omni-research/Tarsier-7b"
model = AutoModelForImageText.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Procesador para manejar imágenes y video (dependiendo de los fotogramas o imágenes que se usen)
processor = AutoProcessor.from_pretrained(model_name)

# Función de inferencia con imágenes (o secuencia de fotogramas)
def inference_with_images(image_folder, question):
    images = []
    # Cargar varias imágenes (o fotogramas de video)
    for img_file in sorted(os.listdir(image_folder)):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_folder, img_file)
            images.append(img_path)
    
    # Formato de la entrada para el modelo
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": images},  # Si es video, puedes hacer algo similar con secuencia de imágenes
                {"type": "text", "text": question},
            ],
        }
    ]
    
    # Preprocesamiento para la inferencia
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = processor.process_images(images)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Realizar la inferencia: Generación de la respuesta
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return output_text

# Función para procesar el dataframe y hacer inferencias para cada pregunta
def process_and_infer(parquet_file, output_csv):
    df = pd.read_parquet(parquet_file)  # Leer archivo parquet

    # Crear un diccionario para almacenar los resultados
    results = []

    # Obtener los IDs de las preguntas
    question_ids = df['question_id'].unique()

    # Barra de progreso para procesar las preguntas
    for question_id in tqdm(question_ids, desc="Procesando preguntas", unit="pregunta"):
        question_data = df[df['question_id'] == question_id]
        segment_id = question_data.iloc[0]['segment_id']
        question = question_data.iloc[0]['question']
        images_folder = os.path.join('images/val', str(segment_id))  # Ruta de las imágenes

        # Inferencia con las imágenes y la pregunta
        answer = inference_with_images(images_folder, question)

        # Almacenar los resultados
        results.append({
            'question_id': question_id,
            'segment_id': segment_id,  # ID de segmento
            'answer': answer  # Respuesta generada
        })

    # Convertir los resultados a DataFrame y exportar a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

# Ejemplo de uso
parquet_file = 'val.parquet'  # Archivo parquet con las preguntas
output_csv = 'predictions.csv'  # Archivo de salida
process_and_infer(parquet_file, output_csv)
