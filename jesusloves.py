# pp.py  — Full fixed version

import json
import os
import traceback
from flask import Flask, render_template, request, jsonify, redirect, url_for, current_app
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from weasyprint import HTML

print(tf.__version__)
print(hub.__version__)


# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------
# Google API Key for Gemini
# -------------------------
# NOTE: keep your own API key secure. Replace below or set GOOGLE_API_KEY in environment.
GOOGLE_API_KEY = "AIzaSyAVPcTFgt1LLZauvKSluzfdA73f51hRF1Q"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set your Google API Key in GOOGLE_API_KEY")

# -------------------------
# Initialize AI Agent
# -------------------------
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# -------------------------
# Queries for Gemini
# -------------------------
skin_query = """
You are a veterinary skin disease expert.
Analyze the uploaded animal image and return only the predicted disease name and its probability (out of 100).
"""

animal_query = """
You are a veterinary expert. Analyze the uploaded animal image.
Return ONLY:
- Animal Name
- Breed
"""

# pp.py  — Add these near the bottom of the file, after your other /predict_* routes

# 1) Prompt template for symptoms → disease
symptoms_query_template = """
You are a veterinary disease expert.
Based on these symptoms: {symptoms},
return ONLY the predicted disease name.
"""

# -------------------------
# Safe import for prescription logic (robust)
# -------------------------
# Try to import the main prescription function from possible locations.
try:
    # Preferred: module generate_prescription_logic.py provides generate_prescription_logic
    from generate_prescription_logic import generate_prescription_logic
    app.logger.info("Imported generate_prescription_logic from generate_prescription_logic1.py")
except Exception:
    try:
        # Fallback: utils.py might provide prescription_logic; adapt name
        from utils import prescription_logic as generate_prescription_logic
        app.logger.info("Imported generate_prescription_logic from utils.prescription_logic")
    except Exception:
        generate_prescription_logic = None
        app.logger.warning("generate_prescription_logic not found. /generate_prescription will return error if USE_STUB=False")

# -------------------------
# Helper Functions for Gemini
# -------------------------
def analyze_image(image_path, query_text):
    """
    Resize image, wrap it into AgnoImage, call medical_agent.run with images,
    return response.content string. Clean up temp resized file.
    """
    temp_path = None
    try:
        image = PILImage.open(image_path).convert("RGB")
        width, height = image.size
        aspect_ratio = width / height if height != 0 else 1
        new_width = 500
        new_height = max(1, int(new_width / aspect_ratio))
        resized_image = image.resize((new_width, new_height))

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "resized.png")
        resized_image.save(temp_path, format="PNG")
        agno_image = AgnoImage(filepath=temp_path)

        response = medical_agent.run(query_text, images=[agno_image])
        # `response` is expected to have a `.content` attribute (string)
        return response.content
    except Exception as e:
        app.logger.exception("Error in analyze_image")
        return f"⚠️ Analysis error: {e}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

# -------------------------
# Load ML Models
# -------------------------
# Make sure these paths exist relative to where you run pp.py

tf_model_path = 'model/20220804-16551659632113-all-images-Adam.h5'
pkl_model_path = 'model/dogModel1.pkl'

custom_objects = {'KerasLayer': hub.KerasLayer}
if not os.path.exists(tf_model_path):
    app.logger.warning("TensorFlow model not found at %s. predict_breed may crash.", tf_model_path)
if not os.path.exists(pkl_model_path):
    app.logger.warning("Joblib model not found at %s. predict_disease may crash.", pkl_model_path)

# Load tf model inside try/except to avoid crash at app startup in dev
model = None
try:
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(tf_model_path, custom_objects=custom_objects)
    app.logger.info("Loaded tensorflow model from %s", tf_model_path)
except Exception as e:
    app.logger.exception("Failed to load tensorflow model: %s", e)
    model = None

joblib_model = None
try:
    joblib_model = joblib.load(pkl_model_path)
    app.logger.info("Loaded joblib model from %s", pkl_model_path)
except Exception as e:
    app.logger.exception("Failed to load joblib model: %s", e)
    joblib_model = None

# -------------------------
# Load & preprocess dataset
# -------------------------
# Load the data and preprocess it
df = pd.read_csv("data/dog_data_09032022.csv")
# Perform data cleaning and preprocessing steps from the provided code
# 5)
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def hwl_clean(height): # height weight life clean
    height = str(height)
    height_list = height.split('-')
    result = []
    for word in height_list:
        result = result + word.split(" ")
    avg_val = 0
    count = 0
    for i in result:
        if is_number(i):
            count = count +1
            avg_val = avg_val + float(i)
    if count != 0:
        avg_val = avg_val / count
        return avg_val
    else:
        return 0

df['height_c'] = df['height'].apply(hwl_clean)
df['weight_c'] = df['weight'].apply(hwl_clean)
df['life_c'] = df['life'].apply(hwl_clean)

# replace the outlier with the mean value
mean_val = (df["life_c"].sum() - df["life_c"].max()) / 275
df["life_c"].replace(df["life_c"].max(), mean_val, inplace= True)

df.drop(['height','weight','life'], axis = 1,inplace = True)

#6)
df[['height_c','weight_c','life_c']].isnull().sum()

#7)
def popular_clean(pop):
    if type(pop) == float:
        return np.nan
    else:
        rank_list = pop.split(" of ")
        measure = int(rank_list[0]) / int(rank_list[1])
        return 1 - measure

df['popularity_rank_c']= df['popularity_rank'].apply(popular_clean)
df.drop('popularity_rank', axis = 1, inplace = True)
df['Coat Length'].value_counts()

#8)
def coatLen_clean(coat):
    if type(coat) != float:
        if "Long" in coat:
            return 3
        elif "Medium" in coat:
            return 2
        else:
            return 1
    else:
        return np.nan

df["Coat_Length_c"] = df["Coat Length"].apply(coatLen_clean)
df.drop("Coat Length", axis = 1, inplace= True)

#9)
coat_dict = {}
for i in df["Coat Type"]:
    if type(i) != float:
        coat_list = i.split('-')
        for coat in coat_list:
            if coat not in coat_dict:
                coat_dict[coat] = 1
            else:
                coat_dict[coat] = coat_dict[coat] + 1

new_dict = dict(sorted(coat_dict.items(), key=lambda item: item[1], reverse= True))
new_dict

#10)
def coat_clean(coat):
    if type(coat) != float:
        if 'Double' in coat:
            return 'Double'
        elif 'Smooth' in coat:
            return 'Smooth'
        elif 'Wiry' in coat:
            return 'Wiry'
        elif 'Silky' in coat:
            return 'Silky'
        elif 'Curly' in coat:
            return 'Curly'
        elif 'Rough' in coat:
            return 'Rough'
        elif 'Corded' in coat:
            return 'Corded'
        elif 'Hairless' in coat:
            return 'Hairless'
    else:
        return np.nan
df['coat_c'] = df['Coat Type'].apply(coat_clean)
df.drop("Coat Type", axis = 1, inplace = True)
df.drop("marking", axis = 1, inplace = True)
df.drop("color", axis = 1, inplace = True)
# Normalized Euclidean columns
euclidean_cols = ['height_c', 'weight_c', 'life_c',
                  'Coat_Length_c', 'Affectionate With Family',
                  'Good With Young Children', 'Good With Other Dogs', 'Shedding Level',
                  'Coat Grooming Frequency', 'Drooling Level', 'Openness To Strangers',
                  'Playfulness Level', 'Watchdog/Protective Nature', 'Adaptability Level',
                  'Trainability Level', 'Energy Level', 'Barking Level',
                  'Mental Stimulation Needs']
df_euclidean = df[euclidean_cols]
df_euclidean.fillna(df_euclidean.mean(), inplace=True)
normalized_df_euclidean = (df_euclidean - df_euclidean.min()) / (df_euclidean.max() - df_euclidean.min())

# Function to get the most similar breeds
def get_names(profile):
    dist_list = []
    for i in range(normalized_df_euclidean.shape[0]):
        dist_list.append(np.linalg.norm(profile - normalized_df_euclidean.iloc[i]))
    idx_list = sorted(range(len(dist_list)), key=lambda i: dist_list[i], reverse=False)[:5]
    names = df['dog'][idx_list].values
    distances = 1 - (sorted(dist_list)[0:5] / sorted(dist_list)[-1])
    return names, distances

@app.route('/recommend_breed', methods=['GET', 'POST'])
def recommend_breed():
    if request.method == 'POST':
        json_data = request.data.decode('utf-8')

        # Parse the JSON data into a Python dictionary
        data = json.loads(json_data)

        # Access each JSON object
        for key, value in data.items():
            print(key, value)
        # Get the form data
        height = float(data['height_c'])
        print("Height", height)
        weight = float(data['weight_c'])
        life = float(data['life_c'])
        coat_length = float(data['coat_length_c'])
        affection = float(data['affection_c'])
        good_with_kids = float(data['good_with_kids_c'])
        good_with_dogs = float(data['good_with_dogs_c'])
        shedding = float(data['shedding_c'])
        grooming = float(data['grooming_c'])
        drooling = float(data['drooling_c'])
        openness = float(data['openness_c'])
        playfulness = float(data['playfulness_c'])
        watchdog = float(data['watchdog_c'])
        adaptability = float(data['adaptability_c'])
        trainability = float(data['trainability_c'])
        energy = float(data['energy_c'])
        barking = float(data['barking_c'])
        mental_stimulation = float(data['mental_stimulation_c'])

        # Normalize the input values
        profile = [(height - 6.5) / 32.0, (weight - 5) / 195, (life - 6.5) / 18,
                   (coat_length - 1) / 3, (affection - 1) / 5, (good_with_kids - 1) / 5,
                   (good_with_dogs - 1) / 5, (shedding - 1) / 5, (grooming - 1) / 5,
                   (drooling - 1) / 5, (openness - 1) / 5, (playfulness - 1) / 5,
                   (watchdog - 1) / 5, (adaptability - 1) / 5, (trainability - 1) / 5,
                   (energy - 1) / 5, (barking - 1) / 5, (mental_stimulation - 1) / 5]

        names, distances = get_names(profile)

        # Generate URLs for each recommended breed
        base_url = 'https://www.akc.org/dog-breeds/'
        data = [{'name': name, 'similarity': distance, 'url': base_url + name.lower().replace(' ', '-')} for name, distance in zip(names, distances)]

        return jsonify(data)


# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    return render_template('hom.html')

@app.route('/goto_disease', methods=['POST'])
def goto_disease():
    return redirect(url_for('disease_page'))

@app.route('/disease_page')
def disease_page():
    return render_template('1.html')

@app.route('/two')
def two_page():
    return render_template('two.html')

@app.route('/symptoms')
def symptoms_page():
    return render_template('index1.html')


@app.route("/index")
def index():
    animal = request.args.get("animal")
    breed = request.args.get("breed")
    return render_template("index1.html", animal=animal, breed=breed)

# -------------------------
# Breed Recommendation
# -------------------------

# -------------------------
# Breed Prediction using CNN
# -------------------------
labels = ["Labrador", "Beagle", "German Shepherd", "Bulldog", "Poodle"]

def predict_breed(image):
    if model is None:
        raise RuntimeError("TensorFlow model is not loaded.")
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.constant(image)
    prediction = model.predict(image).flatten()
    return labels[np.argmax(prediction)]

@app.route('/predict_breed_route', methods=['POST'])
def predict_breed_route():

    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image provided"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    try:
        image = cv2.imread(file_path)
        predicted_breed = predict_breed(image)
    except Exception as e:
        app.logger.exception("Error in predict_breed_route")
        predicted_breed = None
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

    if predicted_breed is None:
        return jsonify({"error": "Breed prediction failed"}), 500
    return jsonify({"predicted_breed": predicted_breed})

def run_prescription_logic():
    # Lazy import already handled above; this function kept for compatibility
    if generate_prescription_logic:
        return generate_prescription_logic()
    else:
        raise RuntimeError("generate_prescription_logic not available")

# -------------------------
# Disease Prediction (joblib)
# -------------------------
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.get_json()
    if not data or 'symptoms' not in data:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms = data['symptoms']
    try:
        input_vector = np.array(symptoms).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Invalid symptom format: {e}"}), 400

    if joblib_model is None:
        return jsonify({"error": "Prediction model not loaded"}), 500

    try:
        prediction = joblib_model.predict(input_vector)
        # if prediction is a vector or probabilities, handle gracefully
        if hasattr(prediction, 'shape') and prediction.size > 1:
            predicted_index = int(np.argmax(prediction))
        else:
            # if it gives direct class index
            predicted_index = int(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else int(prediction)

    except Exception as e:
        app.logger.exception("Error during joblib prediction")
        return jsonify({"error": str(e)}), 500

    disease_mapping = {
        0: ('Tick fever', 'https://www.thevetexpert.com/tick-fever-in-dogs-causes-signs-diagnosis-and-treatment/'),
        1: ('Distemper', 'https://www.akc.org/expert-advice/health/distemper-in-dogs/'),
        2: ('Parvovirus', 'https://www.akc.org/expert-advice/health/parvovirus-what-puppy-owners-need-to-know/'),
        3: ('Hepatitis', 'https://vcahospitals.com/know-your-pet/infectious-canine-hepatitis'),
        4: ('Tetanus', 'https://vcahospitals.com/know-your-pet/tetanus-in-dogs'),
        5: ('Chronic kidney Disease', 'https://www.petmd.com/dog/conditions/kidney/chronic-renal-failure-dogs'),
        6: ('Diabetes', 'https://www.petmd.com/dog/conditions/endocrine/c_dg_diabetes_mellitus'),
        7: ('Gastrointestinal Disease', 'https://www.petmd.com/dog/conditions/digestive/c_multi_gastroenteritis'),
        8: ('Allergies', 'https://www.petmd.com/dog/general-health/dog-allergies'),
        9: ('Gingivitis', 'https://www.petmd.com/dog/conditions/mouth/c_dg_gingivitis'),
        10: ('Cancers', 'https://www.petmd.com/dog/conditions/cancer/c_dg_cancer_general'),
        11: ('Skin Rashes', 'https://www.petmd.com/dog/general-health/dog-skin-allergies-and-rashes')
    }

    predicted_disease, disease_url = disease_mapping.get(predicted_index, ("Unknown Disease", "#"))
    return jsonify({"disease": predicted_disease, "url": disease_url})

# -------------------------
# Gemini-based Symptom route
# -------------------------
@app.route('/predict_symptoms_gemini', methods=['POST'])
def predict_symptoms_gemini():
    data = request.get_json()
    symptoms = data.get('symptoms', []) if data else []
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    prompt = symptoms_query_template.format(symptoms=', '.join(symptoms))
    try:
        response = medical_agent.run(prompt)
        raw = response.content.strip()
        disease = next((line for line in raw.splitlines() if line.strip()), raw)
        return jsonify({"disease": disease})
    except Exception as e:
        app.logger.exception("❌ Error in predict_symptoms_gemini")
        return jsonify({"error": str(e)}), 500

# -------------------------
# Gemini-based Image Analysis (Animal & Breed)
# -------------------------
@app.route("/predict_animal_gemini", methods=["POST"])
def predict_animal_gemini():
    if "image" not in request.files:
        return jsonify({"error": "No file selected"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    raw = analyze_image(file_path, animal_query)
    try:
        os.remove(file_path)
    except Exception:
        pass

    # Log raw response for debugging
    app.logger.info("🔹 RAW GEMINI (animal): %s", raw.replace("\n", " \\n "))

    # Parse out Animal and Breed
    animal_name = "Unknown"
    breed_name = "Unknown"
    for line in raw.splitlines():
        text = line.strip().lstrip("-* ").strip()
        if ":" in text:
            key, val = [s.strip() for s in text.split(":", 1)]
            if key.lower().startswith("animal"):
                animal_name = val
            elif key.lower().startswith("breed"):
                breed_name = val

    return jsonify({"animal": animal_name, "breed": breed_name, "raw": raw})

# -------------------------
# Gemini-based Image Analysis (Skin) — used by frontend (/predict_skin_gemini)
# -------------------------
@app.route("/predict_skin_gemini", methods=["POST"])
def predict_skin_gemini():
    if "image" not in request.files:
        return jsonify({"error": "No file selected"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    raw = analyze_image(file_path, skin_query)
    try:
        os.remove(file_path)
    except Exception:
        pass

    app.logger.info("🔹 RAW GEMINI (skin): %s", raw.replace("\n", " \\n "))

    # Try to extract a succinct disease result (first non-empty line)
    result = next((line for line in raw.splitlines() if line.strip()), raw)
    return jsonify({"result": result, "raw": raw})

# Keep /predict_skin for compatibility (calls predict_skin_gemini)
@app.route("/predict_skin", methods=["POST"])
def predict_skin():
    return predict_skin_gemini()

def generate_pdf(result, petNo):
    """
    Renders the templats_pdf.html with the prescription & diet data,
    generates a PDF via WeasyPrint, saves it to UPLOAD_FOLDER, and
    returns a publicly accessible URL.
    """
    # 1) Render the Jinja2 template to an HTML string
    rendered_html = render_template(
        'templats_pdf.html',
        disease            = result.get('disease', ''),
        prescription_plan  = result['prescription_plan'],
        diet_plan          = result['diet_plan'],
        explanation        = result['explanation']
    )

    # 2) Convert HTML string to PDF bytes
    pdf_bytes = HTML(string=rendered_html).write_pdf()

    # 3) Save the PDF file into uploads folder
    filename    = f"{petNo}.pdf"
    output_dir  = current_app.config['UPLOAD_FOLDER']
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'wb') as f:
        f.write(pdf_bytes)

    # 4) Return the external URL for the created PDF file
    return url_for(
        'static',
        filename = f"uploads/{filename}",
        _external = True
    )

# -------------------------
# Animal & Breed Selection (redirect helpers)
# -------------------------
@app.route("/select_animal", methods=["GET"])
def select_animal():
    animal = request.args.get("animal")
    return redirect(url_for("two_page", animal=animal))

@app.route("/select_breed", methods=["GET"])
def select_breed():
    animal = request.args.get("animal")
    breed = request.args.get("breed")
    return redirect(url_for("symptoms_page", animal=animal, breed=breed))

@app.route('/in')
def show_prediction_page():
    return render_template('in.html')  # or whatever template you want to show

@app.route('/generate_prescription', methods=['POST'])
def generate_prescription():
    data = request.get_json()

    try:
        # 1) Run your Gemini logic
        result = generate_prescription_logic(data)
        app.logger.info("✅ Logic returned: %s", result)

        # 2) Optionally produce a PDF if you have a prescription_plan
        pet_no = data.get(
            'petNo',
            'PET-' + str(int(pd.Timestamp.now().timestamp()))
        )
        if isinstance(result, dict) and 'prescription_plan' in result:
            try:
                result['pdf_url'] = generate_pdf(result, pet_no)
            except Exception:
                app.logger.exception(
                      "Failed to generate PDF — continuing without pdf_url"
                )

        # 3) Return final JSON
        return jsonify(result)

    except Exception as e:
        # Log the full stack trace on the server
        tb = traceback.format_exc()
        app.logger.error("🔥 Exception in generate_prescription:\n%s", tb)

        # Return a clean error response
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)