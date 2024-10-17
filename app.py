import os
import numpy as np
import cv2
import easyocr  # Import EasyOCR
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the EasyOCR reader globally
reader = easyocr.Reader(['en'])  # Specify the languages you want to use

# Function to check allowed file types
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Preprocess the image for OCR and visual analysis
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
    resized_image = cv2.resize(denoised_image, (800, int(800 * (denoised_image.shape[0] / denoised_image.shape[1]))))
    contrast_image = cv2.convertScaleAbs(resized_image, alpha=1.5, beta=30)
    sharpened_image = cv2.filter2D(contrast_image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return sharpened_image

# Extract text from the image using EasyOCR
def extract_text_from_image(image):
    processed_image = preprocess_image(image)
    results = reader.readtext(processed_image)
    detected_text = " ".join([result[1] for result in results])
    return detected_text.upper()  # Normalize to uppercase for consistency

# Analyze color to detect freshness (for fruits, vegetables, etc.)
def analyze_freshness(image):
    # Convert the image to HSV (Hue, Saturation, Value) color space for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the thresholds for detecting fresh vs. spoiled produce
    fresh_color_lower = np.array([25, 40, 40])   # Example threshold for green/yellow (fresh produce)
    fresh_color_upper = np.array([85, 255, 255]) # Adjust based on what you're detecting

    # Create a mask to detect regions that fall within the fresh color range
    fresh_mask = cv2.inRange(hsv_image, fresh_color_lower, fresh_color_upper)
    fresh_pixels = cv2.countNonZero(fresh_mask)

    # Calculate the percentage of fresh pixels in the image
    total_pixels = image.shape[0] * image.shape[1]
    freshness_percentage = (fresh_pixels / total_pixels) * 100

    # Determine freshness based on the percentage of fresh pixels
    if freshness_percentage > 50:  # More than 50% fresh pixels indicates a fresh item
        return "Fresh"
    else:
        return "Spoiled"

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
            # Convert the image to OpenCV format
            in_memory_file = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

            # Analyze the image for text using OCR
            detected_text = extract_text_from_image(image)
            
            # Perform freshness analysis based on visual cues
            freshness_status = analyze_freshness(image)

            return render_template('result.html', raw_text=detected_text, freshness_status=freshness_status)

    return render_template('upload.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
