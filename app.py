import os
import numpy as np
import cv2
import pytesseract  # Import pytesseract
from flask import Flask, render_template, request, jsonify, Response
import re
from datetime import datetime
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

# Load known brands from CSV
def load_known_brands(csv_file):
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded brands: {df}")  # Debugging line
        return set(df['Product Name'].str.upper())
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return set()
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return set()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return set()

# Load brands globally
known_brands = load_known_brands('flipkart_products.csv')

# Function to check allowed file types
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Preprocess the image for OCR
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
    resized_image = cv2.resize(denoised_image, (800, int(800 * (denoised_image.shape[0] / denoised_image.shape[1]))))
    contrast_image = cv2.convertScaleAbs(resized_image, alpha=1.5, beta=30)
    sharpened_image = cv2.filter2D(contrast_image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return sharpened_image

# Function to extract text from the image using Tesseract OCR
def extract_text_from_image(image):
    processed_image = preprocess_image(image)
    # Use pytesseract to extract text
    detected_text = pytesseract.image_to_string(processed_image)
    return detected_text.upper()

# Function to extract details including expiration date, MRP, pack size, and brand name
def extract_details(text):
    date_patterns = [
        r"\b(?:0[1-9]|1[0-2])[- /.](?:19|20)\d\d\b",
        r"\b(?:0[1-9]|[12][0-9]|3[01])[- /.](?:0[1-9]|1[0-2])[- /.](?:19|20)\d\d\b"
    ]
    mrp_pattern = r"\bMRP[\s:]*[₹$]?\d+(\.\d{1,2})?\b"
    pack_size_pattern = r"\b\d+\s*(?:g|kg|ml|l|mL|L|oz|ozs|gms|grams)\b"
    brand_pattern = r"\b[A-Z][A-Za-z\s&]+(?=\s+[\d]+\s*(?:g|kg|ml|l|mL|L|oz|ozs|gms|grams|₹$|MRP))"

    expiry_date, mrp, pack_size, brand = None, None, None, None
    
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        if dates:
            expiry_date = dates[0]

    mrp_matches = re.findall(mrp_pattern, text)
    mrp = mrp_matches[0] if mrp_matches else None

    pack_size_matches = re.findall(pack_size_pattern, text)
    pack_size = pack_size_matches[0] if pack_size_matches else None

    brand_matches = re.findall(brand_pattern, text)
    if brand_matches:
        for match in brand_matches:
            match_upper = match.strip().upper()
            if match_upper in known_brands:
                brand = match_upper
                break

    return {
        'expiry_date': expiry_date,
        'mrp': mrp,
        'pack_size': pack_size,
        'brand': brand
    }

# Analyze expiry date based on detected text
def analyze_expiry_date(image, detected_text):
    details = extract_details(detected_text)
    expiry_date = details['expiry_date']

    if expiry_date:
        expiry_status = f"Expiry Date Detected: {expiry_date}"
    else:
        expiry_status = "No Expiry Date Found"

    return details, expiry_status

# Load the model for freshness detection
model_path = 'vegnet_model.h5'
model = tf.keras.models.load_model(model_path)

class_mapping = {0: "Damaged", 1: "Dried", 2: "Old", 3: "Ripe", 4: "Unripe"}

# Function to predict freshness from a frame
def predict_image(frame):
    img = cv2.resize(frame, (128, 128))
    img_array = np.expand_dims(img, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)
    predicted_label = class_mapping.get(predicted_class[0], "Unknown")
    confidence_score = predictions[0][predicted_class[0]]

    freshness_score = confidence_score * 100
    freshness_status = "Fresh" if confidence_score >= 0.70 else "Questionable Freshness" if confidence_score >= 0.40 else "Not Fresh"

    return predicted_label, confidence_score, freshness_score, freshness_status

# Initialize the video capture variable
cap = None

# Function to generate frames from the video feed
def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)  # Start the camera when generating frames
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Perform freshness detection on the frame
            predicted_label, confidence_score, freshness_score, freshness_status = predict_image(frame)

            # Overlay the predictions on the frame
            cv2.putText(frame, f"Label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Freshness Status: {freshness_status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format as part of an HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        
        if file and allowed_file(file.filename):
            in_memory_file = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

            detected_text = extract_text_from_image(image)
            details, expiry_status = analyze_expiry_date(image, detected_text)

            return render_template('result.html', 
                                   raw_text=detected_text, 
                                   expiry_status=expiry_status,
                                   details=details)

    return render_template('upload.html')

@app.route('/freshness', methods=['GET'])
def freshness():
    return render_template('freshness.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap
    if cap is not None:
        cap.release()  # Release the camera when stopping
        cap = None
    return jsonify({"status": "Camera stopped"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
