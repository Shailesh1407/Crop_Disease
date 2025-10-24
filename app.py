import os
import joblib
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer



# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------
# Load Model + Components
# -----------------------------
MODEL_PATH = "model_artifacts/Corn_ImageTextWeather_Model.joblib"
data = joblib.load(MODEL_PATH)
clf = data["model"]
scaler = data["scaler"]
le = data["label_encoder"]
bert_model = SentenceTransformer(data["bert_model_name"])

# -----------------------------
# Image Feature Extractor (ResNet50)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.eval().to(device)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_image_feature(img_path):
    image = Image.open(img_path).convert('RGB')
    img_tensor = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(img_tensor).squeeze().cpu().numpy()
    return feat

# -----------------------------
# Disease Prediction Function
# -----------------------------
def predict_disease(caption, img_path, temp, humidity, rainfall):
    txt_feat = bert_model.encode([caption], convert_to_numpy=True)
    img_feat = extract_image_feature(img_path).reshape(1, -1)
    weather_feat = scaler.transform([[temp, humidity, rainfall]])
    X = np.hstack([txt_feat, img_feat, weather_feat])
    pred = clf.predict(X)
    return le.inverse_transform(pred)[0]

# -----------------------------
# Risk Assessment Function
# -----------------------------
def assess_disease_risk(temp, humidity, rainfall):
    risk_score = 0
    if 20 <= temp <= 30:
        risk_score += 1
    if humidity > 70:
        risk_score += 1
    if rainfall > 40:
        risk_score += 1

    if risk_score == 0:
        return "🟢 Low Risk — Conditions are not favorable for disease spread."
    elif risk_score == 1:
        return "🟡 Moderate Risk — Slightly favorable conditions detected."
    else:
        return "🔴 High Risk — Weather highly favors disease development."

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        caption = request.form["caption"]
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall"])

        # Save uploaded image
        file = request.files["leaf_image"]
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # Predict disease
        prediction = predict_disease(caption, img_path, temp, humidity, rainfall)

        # Assess risk level
        risk_message = assess_disease_risk(temp, humidity, rainfall)

        return render_template(
            "index.html",
            prediction=prediction,
            caption=caption,
            temperature=temp,
            humidity=humidity,
            rainfall=rainfall,
            risk_message=risk_message,
            img_file=file.filename
        )
    return render_template("index.html")

@app.route("/disease/<name>")
def disease_info(name):
    disease_data = {
        "Black_rot": {
            "title": "Black Rot (Apple)",
            "overview": "Black rot is a fungal disease caused by *Botryosphaeria obtusa*. It affects leaves, fruit, and bark, causing circular black lesions and fruit rot.",
            "symptoms": [
                "Purple to black circular spots on leaves",
                "Shriveled black fruit mummies",
                "Cankers on twigs and branches"
            ],
            "organic": [
                "Prune and destroy infected branches and fruits",
                "Spray neem oil (3%) weekly for prevention",
                "Improve air circulation around trees"
            ],
            "chemical": [
                "Apply Mancozeb 75% WP (Indofil M-45)",
                "Use Captan 50% WP or Carbendazim 50% WP",
                "Recommended brands: Indofil, Tata Rallis, Bayer"
            ],
            "spray": "Spray Mancozeb 0.25% once a week for 2 weeks. Avoid spraying during rain.",
            "alternate": "If infestation is severe, consider crop rotation with pear or peach trees for 1–2 seasons."
        },
        "Cedar_rust": {
            "title": "Cedar Rust (Apple)",
            "overview": "Cedar rust is caused by *Gymnosporangium juniperi-virginianae*, a fungus requiring both apple and cedar trees to complete its life cycle.",
            "symptoms": [
                "Orange or yellow spots on leaves",
                "Gelatinous orange horns on cedar trees",
                "Premature leaf fall in severe infections"
            ],
            "organic": [
                "Remove nearby cedar trees (within 1 km) if possible",
                "Use sulfur-based organic sprays",
                "Encourage beneficial fungi and maintain balanced soil"
            ],
            "chemical": [
                "Spray Propiconazole 25% EC or Difenoconazole 25% EC",
                "Fungicide brands: Tilt, Score, or Orbit",
                "Repeat every 10–14 days during infection season"
            ],
            "spray": "Spray Difenoconazole 0.1% every 14 days during spring and early summer.",
            "alternate": "If rust pressure remains high, consider growing resistant varieties like Liberty or Enterprise apples."
        },
        "Apple_scab": {
            "title": "Apple Scab",
            "overview": "Apple scab is a fungal disease caused by *Venturia inaequalis*, leading to dark velvety lesions on leaves and fruits.",
            "symptoms": [
                "Olive-green to brown spots on leaves and fruit",
                "Leaf curling and premature dropping",
                "Scabby or cracked fruit surface"
            ],
            "organic": [
                "Remove and destroy fallen leaves",
                "Spray compost tea or neem oil every 7–10 days",
                "Use resistant varieties like Priscilla, Liberty"
            ],
            "chemical": [
                "Apply Captan 50% WP or Dodine 65% WP",
                "Use fungicides: Systhane, Delan, or Score",
                "Repeat every 10 days in rainy conditions"
            ],
            "spray": "Spray Captan 0.3% after petal fall, then every 10 days for 3 applications.",
            "alternate": "Grow resistant varieties or switch to pear during rest years."
        },
        "Corn_common_rust": {
        "title": "Common Rust (Corn)",
        "overview": "Common rust is a fungal disease caused by *Puccinia sorghi*, producing reddish-brown pustules on leaves.",
        "symptoms": [
            "Small, circular reddish-brown pustules on leaf surfaces",
            "Premature leaf senescence in severe cases",
            "Reduced photosynthetic area leading to lower yield"
        ],
        "organic": [
            "Rotate crops to break disease cycle",
            "Remove volunteer corn plants",
            "Apply compost or organic fungicides like neem extracts"
        ],
        "chemical": [
            "Use Propiconazole or Azoxystrobin-based fungicides",
            "Apply when pustules first appear",
            "Follow label instructions for dosage and timing"
        ],
        "spray": "Spray fungicide at the early stage of rust appearance, usually 2–3 applications at 10–14 day intervals.",
        "alternate": "Plant rust-resistant corn varieties and maintain proper plant spacing for air circulation."
    },
    "Corn_Northern_Leaf_Blight": {
        "title": "Northern Leaf Blight (Corn)",
        "overview": "Northern leaf blight is caused by *Exserohilum turcicum*, leading to elongated gray-green lesions on leaves.",
        "symptoms": [
            "Long, elliptical gray-green lesions on older leaves",
            "Lesions may coalesce under severe infection",
            "Yield reduction due to decreased photosynthetic area"
        ],
        "organic": [
            "Remove infected plant debris after harvest",
            "Rotate crops with non-host plants",
            "Encourage beneficial microbes in soil"
        ],
        "chemical": [
            "Apply fungicides such as Azoxystrobin or Tebuconazole",
            "Spray when lesions first appear",
            "Follow label instructions for timing and dosage"
        ],
        "spray": "Use preventive fungicide spray at the onset of disease in susceptible fields.",
        "alternate": "Plant resistant hybrids and avoid planting corn after corn in the same field."
    },
    "Corn_healthy": {
        "title": "Healthy Corn",
        "overview": "Healthy corn exhibits vibrant green leaves, upright growth, and absence of disease symptoms.",
        "symptoms": [
            "Uniform green color",
            "No spots, lesions, or discoloration",
            "Strong stalks and good tassel formation"
        ],
        "organic": [
            "Maintain proper soil fertility",
            "Ensure balanced irrigation",
            "Regularly monitor for pests and diseases"
        ],
        "chemical": [
            "No chemical treatment required for healthy plants"
        ],
        "spray": "Not required",
        "alternate": "Continue good agronomic practices to maintain plant health"
    },
    "Apple_healthy": {
        "title": "Healthy Apple",
        "overview": "Healthy apple trees have vibrant green leaves, uniform fruit development, and no signs of disease.",
        "symptoms": [
            "No spots or lesions on leaves",
            "Firm, unblemished fruit",
            "Good flowering and fruit set"
        ],
        "organic": [
            "Maintain proper pruning and hygiene",
            "Apply compost and mulch for soil health",
            "Ensure proper irrigation and fertilization"
        ],
        "chemical": [
            "No fungicide or pesticide treatment needed if healthy"
        ],
        "spray": "Not required",
        "alternate": "Maintain regular monitoring to prevent disease outbreak"
        }
    }

    disease = disease_data.get(name)
    if not disease:
        return "<h2>Disease information not available.</h2>"
    return render_template("disease_info.html", disease=disease)

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
