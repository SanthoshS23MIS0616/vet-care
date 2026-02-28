import os
import io
import logging
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
import tensorflow as tf
import joblib
from google import genai
from google.genai import types
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

# ----- App Initialization -----
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(module)s]: %(message)s"
)
logger = logging.getLogger(__name__)

# ----- Configuration -----
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----- Model Loading (at startup for efficiency) -----
# 1. TensorFlow Keras model for disease prediction
try:
    DISEASE_MODEL_PATH = os.getenv('DISEASE_MODEL_PATH', 'models/disease_model.h5')
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH, compile=False)
    logger.info(f"Disease model loaded from {DISEASE_MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading disease model: {e}")
    disease_model = None

# 2. joblib/scikit-learn model for breed prediction
try:
    BREED_MODEL_PATH = os.getenv('BREED_MODEL_PATH', 'models/breed_model.joblib')
    breed_model = joblib.load(BREED_MODEL_PATH)
    logger.info(f"Breed model loaded from {BREED_MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading breed model: {e}")
    breed_model = None

# 3. List of disease and breed classes (to match indices)
DISEASE_CLASSES = [
    'Canine Parvovirus', 'Feline Calicivirus', 'Rabies', 'Arthritis', 'Diabetes', 'Dermatitis',
    'Leptospirosis', 'Distemper', 'Kennel Cough', 'Mange', 'Ringworm', 'Lyme Disease', 'Panleukopenia',
    # ... (complete as needed)
]

BREED_CLASSES = [
    'Labrador Retriever', 'German Shepherd', 'Golden Retriever', 'Bulldog',
    'Beagle', 'Poodle', 'Rottweiler',
    # ... (complete as needed)
]

# 4. Disease-to-prescription mapping (for demo; in production, use DB/config)
DISEASE_PRESCRIPTIONS = {
    "Canine Parvovirus": {
        "medications": ["IV fluids", "antiemetics", "antibiotics (if secondary)", "nutritional support"],
        "instructions": "Strict hydration; isolate the animal; monitor for sepsis.",
        "followup": "Monitor GI signs daily; reassess in 3 days or sooner if deteriorating."
    },
    "Arthritis": {
        "medications": ["NSAIDs", "joint supplements (glucosamine/chondroitin)"],
        "instructions": "Encourage gentle exercise; avoid obesity; warm bedding.",
        "followup": "Check in 2 weeks to assess pain and mobility."
    },
    "Diabetes": {
        "medications": ["Insulin (dose by weight)", "special diet (high fiber/low fat)"],
        "instructions": "Twice-daily insulin injections; monitor glucose at home.",
        "followup": "Weekly glucose curves until stable."
    },
    "Dermatitis": {
        "medications": ["Topical corticosteroids", "antimicrobial shampoo"],
        "instructions": "Bathe affected area daily; prevent licking.",
        "followup": "Return in 5-7 days for recheck."
    },
    # ... (add others as needed)
}

# ----- Gemini API Setup -----
try:
    genai_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not genai_api_key:
        logger.error("Google Gemini API key not set in environment variable 'GEMINI_API_KEY' or 'GOOGLE_API_KEY'.")
    genai_client = genai.Client(api_key=genai_api_key)
    GEMINI_IMAGE_MODEL = 'gemini-2.5-flash'
    GEMINI_TEXT_MODEL = 'gemini-2.5-flash'
except Exception as e:
    logger.error(f"Cannot initialize Gemini client: {e}")
    genai_client = None

# ----- Utility: PDF Generator for Prescriptions -----
def generate_prescription_pdf(patient_info, disease, prescription):
    """
    patient_info: dict with 'name', 'age', 'species', etc.
    disease: string
    prescription: dict (output from DISEASE_PRESCRIPTIONS)
    """
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    y = 800

    # Title and patient info
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, y, "Veterinary Prescription Sheet") 
    p.setFont("Helvetica", 12)
    y -= 40
    for key, value in patient_info.items():
        p.drawString(100, y, f"{key.capitalize()}: {value}")
        y -= 20

    # Disease and prescription
    y -= 10
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, y, f"Diagnosis: {disease}")
    y -= 30
    p.setFont("Helvetica", 12)
    p.drawString(100, y, "Medications:")
    y -= 20
    for med in prescription.get("medications", []):
        p.drawString(120, y, f"- {med}")
        y -= 20

    p.drawString(100, y, "Instructions:")
    y -= 20
    for line in prescription.get("instructions", "").split('\n'):
        p.drawString(120, y, line)
        y -= 15

    p.drawString(100, y, "Follow-up:")
    y -= 20
    for line in prescription.get("followup", "").split('\n'):
        p.drawString(120, y, line)
        y -= 15

    p.setFont("Helvetica-Oblique", 10)
    y -= 30
    p.drawString(100, y, "Please follow veterinary professional advice. Contact clinic in case of adverse events.")

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# ----- Helper: Secure File Save & Validation -----
def save_uploaded_file(file_storage, allowed_types=None):
    filename = secure_filename(file_storage.filename)
    if not filename:
        raise ValueError("Empty filename.")
    ext = os.path.splitext(filename)[1].lower()
    if allowed_types and ext not in allowed_types:
        raise ValueError("Unsupported file type: " + ext)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(path)
    return path

# ----- ROUTES -----

@app.route('/')
def index():
    return jsonify({"msg": "Veterinary Flask Application: TensorFlow, joblib, Gemini, PDF prescription. Ready."})

@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    """
    Expects:
    - If 'image' in request.files: uses TensorFlow model for image-based prediction.
    - Else: expects JSON {"symptoms": "..."}; uses text-based prediction via Gemini.
    """
    try:
        if 'image' in request.files:
            image_file = request.files['image']
            path = save_uploaded_file(image_file, allowed_types=['.jpg', '.jpeg', '.png', '.webp'])
            img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = x / 255.0
            x = tf.expand_dims(x, axis=0)
            if disease_model is None:
                abort(500, "Disease model not loaded.")
            preds = disease_model.predict(x)
            idx = int(tf.argmax(preds[0]))
            confidence = float(preds[0][idx])
            disease = DISEASE_CLASSES[idx]
            logger.info(f"Image prediction: {disease} (conf: {confidence:.2f})")
            return jsonify({"disease": disease, "confidence": confidence})

        else:
            # Symptoms-based (text) prediction via Gemini or fallback to internal logic
            data = request.get_json(force=True)
            symptoms = data.get("symptoms", None)
            if not symptoms or not genai_client:
                abort(400, "No symptoms provided or Gemini not configured.")
            prompt = f"The following animal symptoms are noted: {symptoms}. List the single most likely disease (do not explain)."
            response = genai_client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=30, temperature=0.2)
            )
            disease = response.text.strip().split('\n')[0]
            logger.info(f"Text prediction: {disease}")
            return jsonify({"disease": disease})

    except Exception as e:
        logger.error(f"/predict/disease error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/breed', methods=['POST'])
def predict_breed():
    """
    Expects an/uploaded image file. Uses joblib scikit-learn model.
    """
    try:
        if 'image' not in request.files:
            abort(400, "'image' file required for breed prediction.")
        image_file = request.files['image']
        path = save_uploaded_file(image_file, allowed_types=['.jpg', '.jpeg', '.png', '.webp'])
        img = tf.keras.preprocessing.image.load_img(path, target_size=(128, 128))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = x / 255.0
        x = x.flatten().reshape(1, -1)  # Depending on model input shape
        if breed_model is None:
            abort(500, "Breed model not loaded.")
        pred = breed_model.predict(x)
        breed = BREED_CLASSES[int(pred[0])]
        logger.info(f"Breed prediction: {breed}")
        return jsonify({"breed": breed})
    except Exception as e:
        logger.error(f"/predict/breed error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/image', methods=['POST'])
def gemini_image_analysis():
    """
    Expects image upload and (optional) prompt as form data.
    Returns Gemini's JSON/textual analysis of image.
    """
    try:
        if not genai_client:
            abort(500, "Gemini not configured.")
        if 'image' not in request.files:
            abort(400, "'image' file required.")
        image_file = request.files['image']
        image_bytes = image_file.read()
        prompt = request.form.get('prompt', "Describe this veterinary image in detail.")
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        response = genai_client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(max_output_tokens=128)
        )
        logger.info(f"Gemini image analysis: {response.text[:100]}...")
        return jsonify({"gemini_analysis": response.text})
    except Exception as e:
        logger.error(f"/analyze/image error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze/symptoms', methods=['POST'])
def gemini_symptom_analysis():
    """
    POST: JSON with 'symptoms' (text).
    Calls Gemini for analysis.
    """
    try:
        if not genai_client:
            abort(500, "Gemini not configured.")
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", None)
        if not symptoms:
            abort(400, "'symptoms' field required.")
        prompt = (
            f"Animal presenting with: {symptoms}\n"
            "1. List the single most likely disease.\n"
            "2. List 2-3 possible alternative diagnoses.\n"
            "3. Suggest a likely treatment protocol."
        )
        response = genai_client.models.generate_content(
            model=GEMINI_TEXT_MODEL, contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=128)
        )
        logger.info(f"Gemini symptom analysis: {response.text[:120]}...")
        return jsonify({"gemini_result": response.text})
    except Exception as e:
        logger.error(f"/analyze/symptoms error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prescriptions', methods=['POST'])
def generate_prescription():
    """
    POST JSON:
    {
      "disease": "...",
      "patient_info": {"name": "...", "age": "...", ...}
    }
    Returns a prescription recommendation from DISEASE_PRESCRIPTIONS.
    """
    try:
        data = request.get_json(force=True)
        disease = data.get("disease", None)
        patient_info = data.get("patient_info", {})
        # Disease lookup
        prescription = DISEASE_PRESCRIPTIONS.get(disease)
        if not prescription:
            # Optionally, try Gemini as fallback
            if genai_client:
                prompt = f"Provide a concise veterinary prescription regimen for {disease} in a clinic setting (table format please)."
                response = genai_client.models.generate_content(
                    model=GEMINI_TEXT_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=80, temperature=0.2)
                )
                logger.info(f"Prescription via Gemini for {disease}: {response.text.strip()}")
                return jsonify({
                    "disease": disease,
                    "prescription": response.text.strip()
                })
            else:
                abort(404, f"No prescription found for {disease}; Gemini fallback unavailable.")
        logger.info(f"Prescription generated for {disease}")
        result = {
            "disease": disease,
            "prescription": prescription,
            "patient_info": patient_info,
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"/prescriptions error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prescriptions/pdf', methods=['POST'])
def prescription_pdf():
    """
    POST JSON:
    {
      "disease": "...",
      "patient_info": {name: ..., age: ..., species: ...}
    }
    Returns: PDF file as attachment.
    """
    try:
        data = request.get_json(force=True)
        disease = data.get("disease", None)
        patient_info = data.get("patient_info", {})
        # Lookup prescription logic
        prescription = DISEASE_PRESCRIPTIONS.get(disease)
        if not prescription:
            abort(404, "Prescription unavailable for this disease.")
        # Generate PDF as bytes buffer
        pdf_buffer = generate_prescription_pdf(patient_info, disease, prescription)
        response = send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"{patient_info.get('name', 'patient')}_prescription.pdf",
            mimetype='application/pdf'
        )
        logger.info("Prescription PDF generated and served.")
        return response
    except Exception as e:
        logger.error(f"/prescriptions/pdf error: {e}")
        return jsonify({"error": str(e)}), 500

# ----- Error Handling -----
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": getattr(error, 'description', str(error))}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": getattr(error, 'description', str(error))}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Internal server error: %s", error)
    return jsonify({"error": "Internal Server Error"}), 500

# ----- Integration Test Setup (basic example) -----
@app.route('/test/ping', methods=['GET'])
def test_ping():
    """Health check for integration testing."""
    return jsonify({"status": "ok"})

# ----- Main -----
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)