import os
import io
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

# --- IMPORT VALIDATOR ---
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobile_preprocess, decode_predictions

# Decorator FixedDropout
@tf.keras.utils.register_keras_serializable()
class FixedDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate, seed=self.seed, noise_shape=self.noise_shape)
        return inputs
    
    def get_config(self):
        config = super(FixedDropout, self).get_config()
        config.update({
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

app = Flask(__name__)
CORS(app)

# Konfigurasi
MODEL_FILENAME = 'model_cnn.keras' 
MODEL_PATH = os.path.join('model', MODEL_FILENAME)
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# --- 1. INISIALISASI MODEL ---
model = None
validator_model = None
class_names = []

try:
    # Load Model Penyakit
    custom_objects = {
        "FixedDropout": FixedDropout,
        "swish": tf.keras.activations.swish
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model Penyakit berhasil dimuat!")
    
    # Load Label
    LABELS_PATH = os.path.join('model', 'labels.txt')
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✅ Label dimuat: {len(class_names)} kelas.")

    # Load Validator
    print("⏳ Memuat Validator (MobileNetV2)...")
    validator_model = MobileNetV2(weights='imagenet')
    print("✅ Validator siap!")

except Exception as e:
    print(f"❌ Error saat memuat model: {e}")

# Mapping Slug
prediction_slugs = {
    "Daun Jagung Terserang Cercospora Leaf Spot Gray Leaf Spot": "Gray_Leaf",
    "Daun Jagung Terserang Common Rust": "Common_Rust",
    "Daun Jagung Terserang Northern Leaf Blight": "Northern_Leaf",
    "Daun Kentang Terserang Early Blight": "Early_Blight",
    "Daun Kentang Terserang Late Blight": "Late_Blight2",
    "Daun Padi Terserang Bacterial Leaf Blight": "Bacterial_Leaf",
    "Daun Padi Terserang Brown Spot": "Brown_Spot",
    "Daun Padi Terserang Leaf Scald": "Leaf_Scald",
    "Daun Padi Terserang Sheath Blight": "Sheath_Blight",
    "Daun Singkong Terserang Bacterial Blight": "Bacterial_Blight",
    "Daun Singkong Terserang Brown Streak Disease": "Brown_Streak",
    "Daun Singkong Terserang Green Mottle": "Green_Mottle",
    "Daun Singkong Terserang Mosaic Disease": "Mosaic_Disease",
    "Daun Tomat Terserang Bacterial Spot": "Bacterial_Spot",
    "Daun Tomat Terserang Late Blight": "Late_Blight",
    "Daun Tomat Terserang Leaf Mold": "Leaf_Mold",
    "Daun Tomat Terserang Septoria Leaf Spot": "Leaf_Spot",
    "Daun Tomat Terserang Yellow Leaf Curl Virus": "Leaf_Curl"
}

# --- 2. LIST KATA KUNCI TANAMAN ---
PLANT_KEYWORDS = [
    'leaf', 'plant', 'flower', 'vegetable', 'fruit', 'tree', 'grass', 
    'corn', 'ear', 'grain', 'wheat', 'rice', 'agriculture', 'garden',
    'pot', 'greenhouse', 'cabbage', 'broccoli', 'cauliflower', 'cucumber',
    'zucchini', 'squash', 'potato', 'daisy', 'rose', 'fungus', 'mushroom'
]

def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- 3. FUNGSI CEK VALIDITAS ---
def is_valid_plant(image_data):
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img = img.resize((224, 224))
        
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = mobile_preprocess(x) 
        
        preds = validator_model.predict(x)
        decoded = decode_predictions(preds, top=5)[0] 
        
        print("🔍 Deteksi Objek Umum:", decoded) 
        
        for _, label, _ in decoded:
            label_lower = label.lower()
            if any(keyword in label_lower for keyword in PLANT_KEYWORDS):
                return True, label 
            
        return False, decoded[0][1] 
        
    except Exception as e:
        print(f"Validator Error: {e}")
        return True, "Error" 

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/scan')
def scan_page():
    return render_template('scan.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Request tidak berisi file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if model is None or validator_model is None:
        return jsonify({'error': 'Model belum siap.'}), 500

    try:
        image_data = file.read()    

        # 1. Cek Apakah Tanaman
        is_plant, detected_object = is_valid_plant(image_data)
        
        if not is_plant:            
            return jsonify({
                'is_not_plant': True,
                'detected_object': detected_object, # <-- TAMBAHKAN INI (Sebelumnya Hilang)
                'message': 'Objek ini sepertinya bukan tanaman.'
            }), 200           

        # 2. Prediksi Penyakit
        processed_image = preprocess_image(image_data)
        prediction = model.predict(processed_image)
        
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(prediction[0])) * 100
        
        predicted_slug = prediction_slugs.get(predicted_class_name, None)

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f"{confidence:.2f}%",
            'prediction_slug': predicted_slug
        })
        
    except Exception as e:
        print(f"Server Error: {e}") # Log error di terminal backend
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)