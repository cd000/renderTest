from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import json
import numpy as np
import tensorflow as tf
import uvicorn

app = FastAPI()

# Load the saved model
model = tf.keras.models.load_model('model.keras')  # or 'custom_cnn_crop_disease_model.h5'

# Load class names
class_names = ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_","Corn_(maize)___healthy","Corn_(maize)___Northern_Leaf_Blight"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = round(100*(np.max(prediction[0])),2)

    # Use the loaded class_names to get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    
    return {"predicted_disease": predicted_class_name, "confidence":confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)