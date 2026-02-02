import os
import uuid
import joblib
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, send_file, make_response
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, auth as admin_auth
import pyrebase
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import csv
from collections import Counter

# Load environment variables
load_dotenv()



# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# -----------------------------
# Firebase Configuration
# -----------------------------
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL", "")
}

# Initialize Pyrebase for client-side authentication
firebase = pyrebase.initialize_app(firebase_config)
auth_client = firebase.auth()

# Initialize Firebase Admin SDK (for Firestore)
db = None
try:
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
    if os.path.exists(service_account_path):
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✓ Firebase Admin SDK initialized successfully with service account")
    else:
        print("⚠ Warning: No service account found. Firestore features will be disabled.")
        print("  To enable history tracking:")
        print("  1. Go to Firebase Console > Project Settings > Service Accounts")
        print("  2. Click 'Generate New Private Key'")
        print("  3. Save as 'serviceAccountKey.json' in project root")
        print("  Authentication will still work with Pyrebase!")
except ValueError:
    # Already initialized
    try:
        db = firestore.client()
    except:
        pass
except Exception as e:
    print(f"⚠ Firebase Admin SDK error: {e}")
    print("  App will run without Firestore (history disabled)")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# Authentication Decorator
# -----------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

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
# Authentication Routes
# -----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        name = request.form.get("name")
        
        print(f"\n=== Signup Attempt ===")
        print(f"Email: {email}")
        print(f"Name: {name}")
        print(f"Password length: {len(password) if password else 0}")
        
        try:
            # Create user with Pyrebase
            print("Attempting to create user with Firebase...")
            user = auth_client.create_user_with_email_and_password(email, password)
            print(f"User created successfully! UID: {user['localId']}")
            
            # Store additional user info in Firestore (if available)
            if db:
                user_data = {
                    "name": name,
                    "email": email,
                    "created_at": datetime.now(),
                    "uid": user['localId']
                }
                db.collection('users').document(user['localId']).set(user_data)
                print("User data saved to Firestore")
            
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            error_message = str(e)
            print(f"\n!!! Signup Error !!!")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {error_message}")
            print(f"Full error: {repr(e)}")
            
            # Parse the error message for better user feedback
            if "EMAIL_EXISTS" in error_message:
                flash('Email already exists. Please login.', 'error')
            elif "WEAK_PASSWORD" in error_message:
                flash('Password is too weak. Please use at least 6 characters.', 'error')
            elif "INVALID_EMAIL" in error_message:
                flash('Invalid email address format.', 'error')
            elif "TOO_MANY_ATTEMPTS" in error_message:
                flash('Too many attempts. Please try again later.', 'error')
            else:
                # Show more specific error for debugging
                flash(f'Error: {error_message}', 'error')
            return render_template("signup.html")
    
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        try:
            # Authenticate user with Pyrebase
            user = auth_client.sign_in_with_email_and_password(email, password)
            
            # Get user data from Firestore (if available)
            if db:
                user_doc = db.collection('users').document(user['localId']).get()
                if user_doc.exists:
                    user_data = user_doc.to_dict()
                else:
                    user_data = {"email": email, "name": email.split('@')[0]}
            else:
                user_data = {"email": email, "name": email.split('@')[0]}
            
            # Store user info in session
            session['user'] = {
                'uid': user['localId'],
                'email': email,
                'name': user_data.get('name', email.split('@')[0]),
                'idToken': user['idToken']
            }
            
            flash('Login successful!', 'success')
            return redirect(url_for('predict'))
        except Exception as e:
            error_message = str(e)
            if "INVALID_PASSWORD" in error_message or "EMAIL_NOT_FOUND" in error_message:
                flash('Invalid email or password.', 'error')
            else:
                flash('Error logging in. Please try again.', 'error')
            return render_template("login.html", error=error_message)
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        caption = request.form["caption"]
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall"])

        # Save uploaded image
        file = request.files.get("leaf_image")
        if file is None or file.filename == "":
            return render_template("index.html", error="Please upload a leaf image.")
        if not is_allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type. Please upload a JPG or PNG image.")

        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(img_path)

        # Predict disease
        prediction = predict_disease(caption, img_path, temp, humidity, rainfall)
        print(f"\n=== Prediction Result ===")
        print(f"Predicted disease: {prediction}")
        print(f"Type: {type(prediction)}")

        # Assess risk level
        risk_message = assess_disease_risk(temp, humidity, rainfall)
        
        # Save prediction to Firebase (if Firestore is available)
        if db:
            try:
                user_uid = session['user']['uid']
                prediction_data = {
                    "user_id": user_uid,
                    "prediction": prediction,
                    "caption": caption,
                    "temperature": temp,
                    "humidity": humidity,
                    "rainfall": rainfall,
                    "risk_message": risk_message,
                    "image_filename": unique_name,
                    "timestamp": datetime.now()
                }
                db.collection('predictions').add(prediction_data)
            except Exception as e:
                print(f"Error saving prediction: {e}")
        else:
            print("Firestore not available - prediction not saved to history")

        return render_template(
            "index.html",
            prediction=prediction,
            caption=caption,
            temperature=temp,
            humidity=humidity,
            rainfall=rainfall,
            risk_message=risk_message,
            img_file=unique_name
        )
    return render_template("index.html")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/crops")
def crops():
    return render_template("crops.html")

@app.route("/diseases")
def diseases():
    return render_template("diseases.html")

@app.route("/prevention")
def prevention():
    return render_template("prevention.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")

@app.route("/history")
@login_required
def history():
    if not db:
        flash('History feature requires Firebase service account. Please set up serviceAccountKey.json', 'warning')
        return render_template("history.html", predictions=[])
    
    try:
        user_uid = session['user']['uid']
        # Get all predictions for the current user (without ordering to avoid index requirement)
        predictions_ref = db.collection('predictions').where('user_id', '==', user_uid).limit(50)
        predictions = predictions_ref.stream()
        
        prediction_list = []
        for pred in predictions:
            pred_data = pred.to_dict()
            pred_data['id'] = pred.id
            prediction_list.append(pred_data)
        
        # Sort by timestamp in Python (descending - newest first)
        prediction_list.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        return render_template("history.html", predictions=prediction_list)
    except Exception as e:
        print(f"History error: {e}")
        flash(f'Error loading history: {str(e)}', 'error')
        return render_template("history.html", predictions=[])

@app.route("/analytics")
@login_required
def analytics():
    if not db:
        flash('Analytics feature requires Firebase service account.', 'warning')
        return render_template("analytics.html", stats=None)
    
    try:
        user_uid = session['user']['uid']
        predictions_ref = db.collection('predictions').where('user_id', '==', user_uid).stream()
        
        predictions = []
        for pred in predictions_ref:
            pred_data = pred.to_dict()
            predictions.append(pred_data)
        
        # Calculate statistics
        total_predictions = len(predictions)
        
        if total_predictions == 0:
            stats = {
                'total': 0,
                'diseases': {},
                'recent': [],
                'risk_distribution': {'Low': 0, 'Moderate': 0, 'High': 0}
            }
        else:
            # Disease distribution
            disease_counts = Counter([p.get('prediction', 'Unknown') for p in predictions])
            
            # Risk distribution
            risk_counts = {'Low': 0, 'Moderate': 0, 'High': 0}
            for p in predictions:
                risk_msg = p.get('risk_message', '')
                if 'Low Risk' in risk_msg:
                    risk_counts['Low'] += 1
                elif 'Moderate Risk' in risk_msg:
                    risk_counts['Moderate'] += 1
                elif 'High Risk' in risk_msg:
                    risk_counts['High'] += 1
            
            # Recent predictions (last 5)
            sorted_predictions = sorted(predictions, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            recent = sorted_predictions[:5]
            
            stats = {
                'total': total_predictions,
                'diseases': dict(disease_counts.most_common()),
                'recent': recent,
                'risk_distribution': risk_counts,
                'avg_temp': sum([p.get('temperature', 0) for p in predictions]) / total_predictions,
                'avg_humidity': sum([p.get('humidity', 0) for p in predictions]) / total_predictions,
                'avg_rainfall': sum([p.get('rainfall', 0) for p in predictions]) / total_predictions
            }
        
        return render_template("analytics.html", stats=stats)
    except Exception as e:
        print(f"Analytics error: {e}")
        flash(f'Error loading analytics: {str(e)}', 'error')
        return render_template("analytics.html", stats=None)

@app.route("/export/pdf")
@login_required
def export_pdf():
    if not db:
        flash('Export feature requires Firebase service account.', 'warning')
        return redirect(url_for('history'))
    
    try:
        user_uid = session['user']['uid']
        predictions_ref = db.collection('predictions').where('user_id', '==', user_uid).stream()
        
        predictions = []
        for pred in predictions_ref:
            pred_data = pred.to_dict()
            predictions.append(pred_data)
        
        predictions.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#10b981'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        elements.append(Paragraph("CropCare AI - Prediction Report", title_style))
        elements.append(Paragraph(f"User: {session['user']['name']}", styles['Normal']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Summary
        summary_style = ParagraphStyle('Summary', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#059669'))
        elements.append(Paragraph("Summary", summary_style))
        elements.append(Paragraph(f"Total Predictions: {len(predictions)}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Predictions table
        if predictions:
            elements.append(Paragraph("Prediction History", summary_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Table data
            data = [['Date', 'Disease', 'Temp (°C)', 'Humidity (%)', 'Rainfall (mm)', 'Risk']]
            
            for pred in predictions[:20]:  # Limit to 20 most recent
                timestamp = pred.get('timestamp', datetime.now())
                date_str = timestamp.strftime('%Y-%m-%d') if isinstance(timestamp, datetime) else 'N/A'
                disease = pred.get('prediction', 'Unknown')[:20]  # Truncate long names
                temp = f"{pred.get('temperature', 0):.1f}"
                humidity = f"{pred.get('humidity', 0):.1f}"
                rainfall = f"{pred.get('rainfall', 0):.1f}"
                risk = 'Low' if 'Low' in pred.get('risk_message', '') else 'Mod' if 'Moderate' in pred.get('risk_message', '') else 'High'
                
                data.append([date_str, disease, temp, humidity, rainfall, risk])
            
            table = Table(data, colWidths=[1.2*inch, 2*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.6*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            elements.append(table)
        
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'cropcare_report_{datetime.now().strftime("%Y%m%d")}.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"PDF Export error: {e}")
        flash(f'Error generating PDF: {str(e)}', 'error')
        return redirect(url_for('history'))

@app.route("/export/csv")
@login_required
def export_csv():
    if not db:
        flash('Export feature requires Firebase service account.', 'warning')
        return redirect(url_for('history'))
    
    try:
        user_uid = session['user']['uid']
        predictions_ref = db.collection('predictions').where('user_id', '==', user_uid).stream()
        
        predictions = []
        for pred in predictions_ref:
            pred_data = pred.to_dict()
            predictions.append(pred_data)
        
        predictions.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Date', 'Time', 'Disease', 'Description', 'Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)', 'Risk Assessment'])
        
        # Data
        for pred in predictions:
            timestamp = pred.get('timestamp', datetime.now())
            date_str = timestamp.strftime('%Y-%m-%d') if isinstance(timestamp, datetime) else 'N/A'
            time_str = timestamp.strftime('%H:%M:%S') if isinstance(timestamp, datetime) else 'N/A'
            
            writer.writerow([
                date_str,
                time_str,
                pred.get('prediction', 'Unknown'),
                pred.get('caption', ''),
                pred.get('temperature', 0),
                pred.get('humidity', 0),
                pred.get('rainfall', 0),
                pred.get('risk_message', '')
            ])
        
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=cropcare_data_{datetime.now().strftime("%Y%m%d")}.csv'
        response.headers['Content-Type'] = 'text/csv'
        
        return response
    except Exception as e:
        print(f"CSV Export error: {e}")
        flash(f'Error generating CSV: {str(e)}', 'error')
        return redirect(url_for('history'))

@app.route("/community")
def community():
    if not db:
        flash('Community feature requires Firebase service account.', 'warning')
        return render_template("community.html", posts=[], alerts=[], stories=[])
    
    try:
        # Get forum posts (without ordering to avoid index requirement)
        posts_ref = db.collection('community_posts').limit(20)
        posts = []
        for post in posts_ref.stream():
            post_data = post.to_dict()
            post_data['id'] = post.id
            posts.append(post_data)
        
        # Sort by timestamp in Python (descending - newest first)
        posts.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        # Get disease alerts (without ordering to avoid index requirement)
        alerts_ref = db.collection('disease_alerts').limit(10)
        alerts = []
        for alert in alerts_ref.stream():
            alert_data = alert.to_dict()
            alert_data['id'] = alert.id
            alerts.append(alert_data)
        
        # Sort by timestamp in Python (descending - newest first)
        alerts.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        # Get success stories (without ordering to avoid index requirement)
        stories_ref = db.collection('success_stories').limit(10)
        stories = []
        for story in stories_ref.stream():
            story_data = story.to_dict()
            story_data['id'] = story.id
            stories.append(story_data)
        
        # Sort by timestamp in Python (descending - newest first)
        stories.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        return render_template("community.html", posts=posts, alerts=alerts, stories=stories)
    except Exception as e:
        print(f"Community error: {e}")
        flash(f'Error loading community: {str(e)}', 'error')
        return render_template("community.html", posts=[], alerts=[], stories=[])

@app.route("/community/post", methods=["POST"])
@login_required
def create_post():
    if not db:
        flash('Community feature requires Firebase service account.', 'warning')
        return redirect(url_for('community'))
    
    try:
        post_type = request.form.get('post_type')  # 'question', 'alert', 'story'
        title = request.form.get('title')
        content = request.form.get('content')
        location = request.form.get('location', '')
        disease = request.form.get('disease', '')
        
        post_data = {
            'user_id': session['user']['uid'],
            'user_name': session['user']['name'],
            'title': title,
            'content': content,
            'location': location,
            'disease': disease,
            'timestamp': datetime.now(),
            'replies': 0
        }
        
        # Save to appropriate collection
        if post_type == 'alert':
            db.collection('disease_alerts').add(post_data)
            flash('Disease alert posted successfully!', 'success')
        elif post_type == 'story':
            db.collection('success_stories').add(post_data)
            flash('Success story shared successfully!', 'success')
        else:
            db.collection('community_posts').add(post_data)
            flash('Post created successfully!', 'success')
        
        return redirect(url_for('community'))
    except Exception as e:
        print(f"Create post error: {e}")
        flash(f'Error creating post: {str(e)}', 'error')
        return redirect(url_for('community'))

@app.route("/community/post/<post_id>")
def view_post(post_id):
    if not db:
        flash('Community feature requires Firebase service account.', 'warning')
        return redirect(url_for('community'))
    
    try:
        # Try to find the post in all collections
        post = None
        post_type = None
        
        # Check community_posts
        post_ref = db.collection('community_posts').document(post_id).get()
        if post_ref.exists:
            post = post_ref.to_dict()
            post['id'] = post_id
            post_type = 'post'
        else:
            # Check disease_alerts
            alert_ref = db.collection('disease_alerts').document(post_id).get()
            if alert_ref.exists:
                post = alert_ref.to_dict()
                post['id'] = post_id
                post_type = 'alert'
            else:
                # Check success_stories
                story_ref = db.collection('success_stories').document(post_id).get()
                if story_ref.exists:
                    post = story_ref.to_dict()
                    post['id'] = post_id
                    post_type = 'story'
        
        if not post:
            flash('Post not found.', 'error')
            return redirect(url_for('community'))
        
        # Get replies (without ordering to avoid index requirement)
        replies_ref = db.collection('post_replies').where('post_id', '==', post_id).stream()
        replies = []
        for reply in replies_ref:
            reply_data = reply.to_dict()
            reply_data['id'] = reply.id
            replies.append(reply_data)
        
        # Sort by timestamp in Python (ascending - oldest first)
        replies.sort(key=lambda x: x.get('timestamp', datetime.min))
        
        return render_template("post_detail.html", post=post, post_type=post_type, replies=replies)
    except Exception as e:
        print(f"View post error: {e}")
        flash(f'Error loading post: {str(e)}', 'error')
        return redirect(url_for('community'))

@app.route("/community/reply/<post_id>", methods=["POST"])
@login_required
def reply_post(post_id):
    if not db:
        flash('Community feature requires Firebase service account.', 'warning')
        return redirect(url_for('community'))
    
    try:
        content = request.form.get('content')
        
        reply_data = {
            'post_id': post_id,
            'user_id': session['user']['uid'],
            'user_name': session['user']['name'],
            'content': content,
            'timestamp': datetime.now()
        }
        
        db.collection('post_replies').add(reply_data)
        
        # Update reply count
        for collection in ['community_posts', 'disease_alerts', 'success_stories']:
            post_ref = db.collection(collection).document(post_id)
            post_doc = post_ref.get()
            if post_doc.exists:
                current_replies = post_doc.to_dict().get('replies', 0)
                post_ref.update({'replies': current_replies + 1})
                break
        
        flash('Reply posted successfully!', 'success')
        return redirect(url_for('view_post', post_id=post_id))
    except Exception as e:
        print(f"Reply error: {e}")
        flash(f'Error posting reply: {str(e)}', 'error')
        return redirect(url_for('view_post', post_id=post_id))

@app.route("/community/delete/<post_id>", methods=["POST"])
@login_required
def delete_post(post_id):
    if not db:
        flash('Community feature requires Firebase service account.', 'warning')
        return redirect(url_for('community'))
    
    try:
        user_uid = session['user']['uid']
        post_deleted = False
        
        # Try to find and delete the post from all collections
        for collection in ['community_posts', 'disease_alerts', 'success_stories']:
            post_ref = db.collection(collection).document(post_id)
            post_doc = post_ref.get()
            
            if post_doc.exists:
                post_data = post_doc.to_dict()
                
                # Check if the user owns this post
                if post_data.get('user_id') == user_uid:
                    # Delete all replies to this post
                    replies_ref = db.collection('post_replies').where('post_id', '==', post_id).stream()
                    for reply in replies_ref:
                        db.collection('post_replies').document(reply.id).delete()
                    
                    # Delete the post
                    post_ref.delete()
                    post_deleted = True
                    flash('Post deleted successfully!', 'success')
                    break
                else:
                    flash('You can only delete your own posts.', 'error')
                    return redirect(url_for('view_post', post_id=post_id))
        
        if not post_deleted:
            flash('Post not found.', 'error')
        
        return redirect(url_for('community'))
    except Exception as e:
        print(f"Delete post error: {e}")
        flash(f'Error deleting post: {str(e)}', 'error')
        return redirect(url_for('community'))

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # Handle contact form submission
        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")
        # In a real application, you would send an email or save to database
        return render_template("contact.html", success=True, message="Thank you for your message! We'll get back to you soon.")
    return render_template("contact.html")

@app.route("/disease/<name>")
def disease_info(name):
    # Normalize the disease name (replace spaces with underscores)
    normalized_name = name.replace(" ", "_")
    
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
    "Northern_Leaf_Blight": {
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

    # Try to find the disease with normalized name
    disease = disease_data.get(normalized_name)
    
    # If not found, try the original name
    if not disease:
        disease = disease_data.get(name)
    
    # If still not found, show error with available diseases
    if not disease:
        print(f"Disease not found: {name} (normalized: {normalized_name})")
        print(f"Available diseases: {list(disease_data.keys())}")
        flash(f'Disease information not available for: {name}', 'warning')
        return redirect(url_for('diseases'))
    
    return render_template("disease_info.html", disease=disease)

@app.errorhandler(404)
def page_not_found(_):
    return render_template("404.html"), 404

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
