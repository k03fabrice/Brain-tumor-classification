from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import tensorflow as tf
import numpy as np
import os

from pytorch_model import PyTorchModel

app = Flask(__name__)

# Modèles globaux
pytorch_model = None
tensorflow_model = None

# Classes du modèle (brain tumor)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor','Pituitary']

# Prétraitement pour PyTorch
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_models():
    """Charge les modèles PyTorch et TensorFlow."""
    global pytorch_model, tensorflow_model

    try:
        print(" Chargement du modèle PyTorch...")
        pytorch_model = PyTorchModel(num_classes=4)
        pytorch_model.load_state_dict(torch.load('models/Fabrice_model.torch', map_location=torch.device('cpu')))
        pytorch_model.eval()
        print(" Modèle PyTorch chargé.")
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle PyTorch : {str(e)}")

    try:
        print(" Loading the TensorFlow model...")
        tensorflow_model = tf.keras.models.load_model('models/Fabrice_model.tensorflow')
        print(" Modèle TensorFlow chargé.")
    except Exception as e:
        raise RuntimeError(f"Error loading TensorFlow model : {str(e)}")

def preprocess_image(image, framework):
    """Prepare an image according to the chosen framework."""
    if framework == 'pytorch':
        return transform(image).unsqueeze(0)
    else:  # tensorflow
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        return np.expand_dims(image, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        model_type = request.form.get('model_type', 'pytorch')

        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'})

        try:
            image = Image.open(file.stream).convert('RGB')

            if model_type == 'pytorch':
                input_tensor = preprocess_image(image, 'pytorch')
                with torch.no_grad():
                    output = pytorch_model(input_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = torch.argmax(prob).item()
                confidence = float(prob[predicted_class])
            elif model_type == 'tensorflow':
                input_tensor = preprocess_image(image, 'tensorflow')
                predictions = tensorflow_model.predict(input_tensor)
                predicted_class = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][predicted_class])
            else:
                return jsonify({'error': f'Unkown model : {model_type}'})

            return jsonify({
                'prediction': CLASS_NAMES[predicted_class],
                'confidence': round(confidence * 100, 2)
            })

        except Exception as e:
            return jsonify({'error': f'Prediction error : {str(e)}'})

    return render_template('index.html')

if __name__ == '__main__':
    try:
        print(" Launching the application...")
        load_models()
        print(" Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models : {str(e)}")
    app.run(debug=True, port=5001)
