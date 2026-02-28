import json
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
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

# -------------------------
# Flask App & Upload Folder
# -------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------------------
# Google API Key for Gemini
# -------------------------
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
# Gemini Queries
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

# -------------------------
# Helper: analyze_image()
# -------------------------
def analyze_image(image_path, query_text):
    temp_path = None
    try:
        image = PILImage.open(image_path)
        width, height = image.size
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height))

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "resized.png")
        resized_image.save(temp_path)
        agno_image = AgnoImage(filepath=temp_path)

        response = medical_agent.run(query_text, images=[agno_image])
        return response.content
    except Exception as e:
        return f"⚠️ Analysis error: {e}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------
# Load ML Models
# -------------------------
path = 'model/20220804-16551659632113-all-images-Adam.h5'
custom_objects = {'KerasLayer': hub.KerasLayer}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(path, custom_objects=custom_objects)

joblib_model = joblib.load('model/dogModel1.pkl')

# -------------------------
# Load & preprocess dataset
# -------------------------
df = pd.read_csv("data/dog_data_09032022.csv")

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def hwl_clean(height):
    height = str(height)
    height_list = height.split('-')
    result = []
    for word in height_list:
        result += word.split(" ")
    nums = [float(x) for x in result if is_number(x)]
    return sum(nums)/len(nums) if nums else 0

df['height_c'] = df['height'].apply(hwl_clean)
df['weight_c'] = df['weight'].apply(hwl_clean)
df['life_c']   = df['life'].apply(hwl_clean)

mean_val = (df["life_c"].sum() - df["life_c"].max()) / (len(df)-1)
df.loc[:, "life_c"] = df["life_c"].replace(df["life_c"].max(), mean_val)
df.drop(['height', 'weight', 'life'], axis=1, inplace=True)

def popular_clean(pop):
    if isinstance(pop, float):
        return np.nan
    num, denom = pop.split(" of ")
    return 1 - (int(num) / int(denom))

df['popularity_rank_c'] = df['popularity_rank'].apply(popular_clean)
df.drop('popularity_rank', axis=1, inplace=True)

def coatLen_clean(coat):
    if isinstance(coat, str):
        if "Long" in coat:   return 3
        if "Medium" in coat: return 2
        return 1
    return np.nan

df["Coat_Length_c"] = df["Coat Length"].apply(coatLen_clean)
df.drop("Coat Length", axis=1, inplace=True)

def coat_clean(coat):
    if isinstance(coat, str):
        for t in ['Double','Smooth','Wiry','Silky','Curly','Rough','Corded','Hairless']:
            if t in coat: return t
    return np.nan

df['coat_c'] = df['Coat Type'].apply(coat_clean)
df.drop(["Coat Type", "marking", "color"], axis=1, inplace=True)

euclidean_cols = [
    'height_c', 'weight_c', 'life_c', 'Coat_Length_c',
    'Affectionate With Family', 'Good With Young Children',
    'Good With Other Dogs', 'Shedding Level', 'Coat Grooming Frequency',
    'Drooling Level', 'Openness To Strangers', 'Playfulness Level',
    'Watchdog/Protective Nature', 'Adaptability Level',
    'Trainability Level', 'Energy Level', 'Barking Level',
    'Mental Stimulation Needs'
]

df_euclidean = df[euclidean_cols].copy()
df.loc[:, "life_c"] = df["life_c"].replace(df["life_c"].max(), mean_val)

normalized_df_euclidean = (
    df_euclidean - df_euclidean.min()
) / (df_euclidean.max() - df_euclidean.min())

def get_names(profile):
    dist_list = [
        np.linalg.norm(profile - normalized_df_euclidean.iloc[i])
        for i in range(normalized_df_euclidean.shape[0])
    ]
    idx_list = sorted(range(len(dist_list)), key=lambda i: dist_list[i])[:5]
    names = df['dog'].values[idx_list]
    distances = 1 - (sorted(dist_list)[:5] / sorted(dist_list)[-1])
    return names, distances

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
    return render_template('index.html')

@app.route('/index')
def index():
    animal = request.args.get("animal")
    breed  = request.args.get("breed")
    return render_template("index.html", animal=animal, breed=breed)

@app.route('/recommend_breed', methods=['POST'])
def recommend_breed():
    data = request.get_json()
    profile = [
        (float(data[k]) - 1)/5
        if k not in ['height_c','weight_c','life_c','coat_length_c']
        else float(data[k])
        for k in data
    ]
    names, distances = get_names(profile)
    base_url = 'https://www.akc.org/dog-breeds/'
    result = [
        {'name': name, 'similarity': dist, 'url': base_url + name.lower().replace(' ','-')}
        for name, dist in zip(names, distances)
    ]
    return jsonify(result)

labels = ["Labrador", "Beagle", "German Shepherd", "Bulldog", "Poodle"]

def predict_breed(image):
    img = cv2.resize(image, (224,224))
    img = np.expand_dims(img, axis=0)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.constant(img)
    preds = model.predict(img).flatten()
    return labels[np.argmax(preds)]

@app.route('/predict_breed_route', methods=['POST'])
def predict_breed_route():
    file = request.files.get('image')
    if not file:
        return {"error": "No image provided"}, 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    img = cv2.imread(path)
    breed = predict_breed(img)
    os.remove(path)
    return {"predicted_breed": breed}

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    data = request.get_json()
    symptoms = data['symptoms']
    input_vector = np.array(symptoms).reshape(1, -1)
    prediction = joblib_model.predict(input_vector)
    idx = np.argmax(prediction)
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
    disease, url = disease_mapping.get(idx, ("Unknown Disease", "#"))
    return {"disease": disease, "url": url}

@app.route("/predict_skin_gemini", methods=["POST"])
def predict_skin_gemini():
    if "image" not in request.files:
        return jsonify(error="No file selected"), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify(error="No file selected"), 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    result = analyze_image(path, skin_query)
    os.remove(path)
    return jsonify(result=result)

@app.route("/predict_animal_gemini", methods=["POST"])
def predict_animal_gemini():
    if "image" not in request.files:
        return jsonify(error="No file selected"), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify(error="No file selected"), 400
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    raw = analyze_image(path, animal_query)
    os.remove(path)  
    print("🔹 RAW GEMINI:", raw.replace("\n"," \\n "))
    animal_name = "Unknown"
    breed_name  = "Unknown"
    for line in raw.splitlines():
        text = line.strip().lstrip("-* ").strip()
        if ":" in text:
            key, val = [s.strip() for s in text.split(":",1)]
            if key.lower().startswith("animal"):
                animal_name = val
            elif key.lower().startswith("breed"):
                breed_name = val
    return jsonify(animal=animal_name, breed=breed_name, raw=raw)

@app.route("/select_animal", methods=["GET"])
def select_animal():
    animal = request.args.get("animal")
    return redirect(url_for("two_page", animal=animal))

@app.route("/select_breed", methods=["GET"])
def select_breed():
    animal = request.args.get("animal")
    breed  = request.args.get("breed")
    # redirect into /index, which reads the two args and injects them into the template
    return redirect(url_for("index", animal=animal, breed=breed))

# -------------------------
# New: in.html & PDF Generation
# -------------------------
@app.route('/in')
def in_page():
    return render_template('in.html')

@app.route('/generate_prescription', methods=['POST'])
def generate_prescription():
    record = request.get_json()

    prompt = f"""
You are a veterinary expert. Given this pet record:
{json.dumps(record)}

Generate and return ONLY valid JSON with these keys:

1) "prescription_plan": list of items with:
   - date, time, medicine, dosage, route, duration, notes

2) "diet_plan": list of items with:
   - date, feeding_time, food_type, quantity, notes

3) "explanation": a clear, owner-friendly summary.

Return your answer as JSON only.
"""

    raw = medical_agent.run(prompt).content
    plan = json.loads(raw)

    pdf_html = render_template('prescription_pdf.html', **plan)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prescriptions.pdf')
    HTML(string=pdf_html).write_pdf(pdf_path)

    return jsonify({
        'prescription_plan': plan['prescription_plan'],
        'diet_plan':        plan['diet_plan'],
        'explanation':      plan['explanation'],
        'pdf_url': url_for('static', filename='uploads/prescriptions.pdf')
    })

# -------------------------
# Run App
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)