from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Nombre del modelo
model_name = "omni-research/Tarsier-7b"

# Descargar el modelo, el tokenizador y el procesador
print("Descargando el modelo, tokenizador y procesador...")
AutoModelForCausalLM.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)
AutoProcessor.from_pretrained(model_name)

print("Modelo, tokenizador y procesador descargados y guardados en la ubicación predeterminada.")
