from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image  # Import the Image module
import requests  # For fetching the image from the URL

# Load the model and processor
model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Define the prompt and image URL
prompt = "What you see into image?"
image_file = "https://cdn.bigdutchman.com/fileadmin/content/egg-poultry/press/news/photos/2024/Feuchtigkeitsmanagement_BD_Asia/Haehnchenmast-broiler-growing-Heukorb-Big-Dutchman_72.jpg"

# Open the image using PIL
raw_image = Image.open(requests.get(image_file, stream=True).raw)

# Preprocess the input
inputs = processor(prompt, raw_image, return_tensors="pt")

# Generate the output
output = model.generate(**inputs, max_new_tokens=20)

# Decode and print the result
print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
