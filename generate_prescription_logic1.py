import os
import json
import traceback
from datetime import datetime, timedelta

# Vertex AI imports (may raise if not installed/configured)
try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
except Exception:
    vertexai = None
    GenerativeModel = None

# Optional PDF helper
try:
    from pdf_utils import generate_pdf as pdf_generate  # type: ignore
except Exception:
    pdf_generate = None

def _safe_json_load(raw_text):
    raw = raw_text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    first = raw.find('{')
    last = raw.rfind('}')
    if first != -1 and last != -1 and last > first:
        candidate = raw[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    raise ValueError("Could not parse JSON from model output.")

def _validate_result_schema(obj):
    if not isinstance(obj, dict):
        return False, "Result is not a JSON object"
    if 'disease' not in obj or not isinstance(obj['disease'], str):
        return False, "Missing or invalid 'disease'"
    if 'prescription_plan' not in obj or not isinstance(obj['prescription_plan'], list):
        return False, "Missing or invalid 'prescription_plan'"
    if 'diet_plan' not in obj or not isinstance(obj['diet_plan'], list):
        return False, "Missing or invalid 'diet_plan'"
    if 'explanation' not in obj or not isinstance(obj['explanation'], str):
        return False, "Missing or invalid 'explanation'"

    if len(obj['prescription_plan']) < 3:
        return False, "Must have at least 3 prescription entries"
    if len(obj['diet_plan']) < 3:
        return False, "Must have at least 3 diet entries"

    for ent in obj['prescription_plan']:
        if not isinstance(ent, dict):
            return False, "prescription_plan entries must be objects"
    for ent in obj['diet_plan']:
        if not isinstance(ent, dict):
            return False, "diet_plan entries must be objects"

    return True, None

def _pad_to_minimum(plan, min_length=3, is_prescription=True):
    """Pad plans to minimum 3 entries with supportive care if short."""
    while len(plan) < min_length:
        base = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "time": "20:00",  # Default evening
            "medicine": "Supportive Care (Rest & Monitoring)",
            "dosage": "As per vet",
            "route": "general",
            "duration": "Ongoing",
            "notes": "Monitor symptoms and consult vet"
        } if is_prescription else {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "feeding_time": "evening",
            "food_type": "Bone Broth",
            "quantity": "50 ml",
            "notes": "Nutrient support for recovery"
        }
        plan.append(base)
    return plan

def _fallback_prescription(disease_keyword, context):
    """Ultra-generic fallback - No fixed mappings. Always supportive, tailored to animal/disease in notes."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    morning = ist_time.replace(hour=8, minute=0).strftime("%H:%M")
    afternoon = ist_time.replace(hour=14, minute=0).strftime("%H:%M")
    evening = ist_time.replace(hour=20, minute=0).strftime("%H:%M")

    animal_note = f" (tailor for {context['animal']} if ruminant/herbivore)" if context['animal'] != "Unknown" else ""
    disease_note = f" for {disease_keyword}" if disease_keyword else ""

    generic_plan = {
        "prescription_plan": [
            {"date": today, "time": morning, "medicine": "Supportive Fluids (Electrolyte Solution)", "dosage": "50-100 ml/kg oral/IV", "route": "oral/injectable", "duration": "Until hydrated", "notes": "Prevent dehydration{0}; vet monitor{1}".format(disease_note, animal_note)},
            {"date": today, "time": afternoon, "medicine": "Multivitamin Injection", "dosage": "5-10 ml IM", "route": "injectable", "duration": "7 days", "notes": "Boost immunity{0}; animal-specific{1}".format(disease_note, animal_note)},
            {"date": today, "time": evening, "medicine": "Probiotic Supplement", "dosage": "1 dose oral", "route": "oral", "duration": "10 days", "notes": "Support gut health{0} recovery{1}".format(disease_note, animal_note)},
            {"date": today, "time": "Daily", "medicine": "Pain Relief (NSAID if fever/pain)", "dosage": "As per vet (e.g., 1.1 mg/kg)", "route": "oral/injectable", "duration": "3-5 days", "notes": "Only if symptoms present; monitor closely for {0}".format(disease_keyword or "the disease")}
        ],
        "diet_plan": [
            {"date": today, "feeding_time": "morning", "food_type": "High-Fiber Soft Feed or Hay", "quantity": "100-200g (pet) or 5-7kg (cow)", "notes": "Easily digestible base for {0}; adjust for animal{1}".format(disease_keyword or "recovery", animal_note)},
            {"date": today, "feeding_time": "afternoon", "food_type": "Nutrient-Rich Broth or Pellets", "quantity": "75-150g (pet) or 3-4kg (cow)", "notes": "Energy boost for {0}; add electrolytes{1}".format(disease_keyword or "recovery", animal_note)},
            {"date": today, "feeding_time": "evening", "food_type": "Protein Supplement (Pellets or Lean Source)", "quantity": "50-100g (pet) or 1-2kg (cow)", "notes": "{0} recovery support; probiotic-mixed{1}".format(disease_keyword or "General", animal_note)},
            {"date": today, "feeding_time": "Daily", "food_type": "Clean Water + Mineral Supplements", "quantity": "Ad lib", "notes": "Essential for hydration in {0}; monitor intake daily{1}".format(disease_keyword or "recovery", animal_note)}
        ],
        "explanation": "For {0}, this is supportive care focused on hydration, nutrition, and preventing complications—no specific cure. Tailor to {1} type (e.g., herbivore for cows). Isolate pet, control vectors (flies/insects if applicable). Monitor daily (temp, appetite)—consult vet immediately for hands-on treatment. This is AI-generated; not a substitute for professional care.".format(disease_keyword or "the disease", context['animal'])
    }
    generic_plan["prescription_plan"] = _pad_to_minimum(generic_plan["prescription_plan"], 4, True)
    generic_plan["diet_plan"] = _pad_to_minimum(generic_plan["diet_plan"], 4, False)
    return {
        "disease": context.get("disease") or (disease_keyword or "Unknown").title(),
        **generic_plan
    }

def generate_prescription_logic(data):
    context = {
        "disease": (data.get("disease") or "").strip(),  # From 'guess' in pp1.py
        "symptoms": data.get("symptoms", []),
        "breed": data.get("breed", ""),
        "age": data.get("age", ""),
        "weight": data.get("weight", ""),
        "sex": data.get("sex", ""),
        "medical_history": data.get("medical_history", ""),
        "petNo": data.get("petNo"),
        "animal": data.get("animal", "Unknown"),
        "petName": data.get("petName", "Unnamed")
    }
    disease_text = context["disease"]
    if not disease_text:
        raise ValueError("No disease provided for prescription generation")

    # Enhanced prompt - Dynamic, disease-specific, forces real-time generation
    example_json = json.dumps({
        "disease": disease_text,  # Dynamic disease
        "prescription_plan": [
            {"date": datetime.utcnow().strftime("%Y-%m-%d"), "time": "08:00", "medicine": "Tailored Med 1 for {0}".format(disease_text), "dosage": "1.1 mg/kg IV", "route": "injectable", "duration": "3-5 days", "notes": "Specific to {0}; vet administer".format(disease_text)},
            {"date": datetime.utcnow().strftime("%Y-%m-%d"), "time": "14:00", "medicine": "Antibiotic for {0}".format(disease_text), "dosage": "20 mg/kg IM", "route": "injectable", "duration": "5 days", "notes": "Prevent infections in {0}".format(disease_text)},
            {"date": datetime.utcnow().strftime("%Y-%m-%d"), "time": "20:00", "medicine": "Vitamin Support for {0}".format(disease_text), "dosage": "5-10 ml IM", "route": "injectable", "duration": "7 days", "notes": "Boost immunity for {0}".format(disease_text)},
            {"date": datetime.utcnow().strftime("%Y-%m-%d"), "time": "Daily", "medicine": "Probiotic for {0}".format(disease_text), "dosage": "1 dose oral", "route": "oral", "duration": "10 days", "notes": "Gut health for {0} recovery".format(disease_text)}
        ],
        "diet_plan": [
            {"date": datetime.utcnow().strftime("%Y-%m-%d"), "feeding_time": "morning", "food_type": "Diet for {0} - Soft Hay/Feed".format(disease_text), "quantity": "100-200g or 5-7kg", "notes": "Tailored base for {0}".format(disease_text)},
            {"date": datetime.utcnow().strftime("%Y-%m-%d"), "feeding_time": "afternoon", "food_type": "Broth/Pellets for {0}".format(disease_text), "quantity": "75-150g or 3-4kg", "notes": "Energy for {0}; electrolytes".format(disease_text)},
            {"date": datetime.utcnow().strftime("%Y-%m-%d"), "feeding_time": "evening", "food_type": "Protein for {0}".format(disease_text), "quantity": "50-100g or 1-2kg", "notes": "Recovery for {0}; probiotic".format(disease_text)},
            {"date": datetime.utcnow().strftime("%Y-%m-%d"), "feeding_time": "Daily", "food_type": "Water/Minerals for {0}".format(disease_text), "quantity": "Ad lib", "notes": "Hydration for {0}; monitor".format(disease_text)}
        ],
        "explanation": "Tailored plan for {0}: Supportive care with hydration, nutrition, and monitoring. No cure—prevent complications. Isolate, control vectors. Consult vet for hands-on. AI only.".format(disease_text)
    }, indent=2)

    prompt = f"""
You are a veterinary AI assistant specializing in pet prescriptions. Based on the disease '{disease_text}', provide a biologically accurate JSON treatment plan with EXACTLY at least 4 medicines and 4 diet plans tailored SPECIFICALLY to this disease. Analyze the disease and generate real-time, disease-specific recommendations (e.g., antivirals for viral, topicals for skin). If skin-related, include 2+ ointments. Use pet profile for context but prioritize disease.

Pet Profile:
- Animal: {context['animal']}
- Breed: {context['breed'] or 'Unknown'}
- Pet Name: {context['petName']}
- Pet Number: {context['petNo']}

EXAMPLE (adapt to '{disease_text}'):
{example_json}

Return ONLY valid JSON—no text outside. Ensure 4+ entries each. If short, add supportive but disease-relevant.

Keys:
- Prescription: date (YYYY-MM-DD), time (HH:MM or 'Daily'), medicine, dosage (mg/kg), route, duration, notes (vet disclaimer).
- Diet: date, feeding_time (morning/afternoon/evening or 'Daily'), food_type, quantity, notes (animal-adjusted).
- Explanation: Concise, disease-focused, with vet emphasis.
- Dates: {datetime.utcnow().strftime("%Y-%m-%d")}. Times: IST 08:00, 14:00, 20:00. Dosages: mg/kg for 10-20kg pet or 200-500kg animal.
"""

    # Primary: Vertex AI (Gemini) for real-time
    if GenerativeModel is not None:
        try:
            vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
            model = GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            raw_text = response.text
            result = _safe_json_load(raw_text)
            is_valid, error = _validate_result_schema(result)
            if is_valid:
                result["prescription_plan"] = _pad_to_minimum(result["prescription_plan"], 4, True)
                result["diet_plan"] = _pad_to_minimum(result["diet_plan"], 4, False)
                if pdf_generate is not None:
                    pdf_path = pdf_generate(result, context)
                    result["pdf_path"] = pdf_path
                return result
            else:
                print(f"Gemini validation failed: {error}. Falling back.")
        except Exception as e:
            print(f"Gemini error: {e}")
            traceback.print_exc()

    # Always fallback to generic (no mapping—ensures variety only from Gemini)
    print("Using generic fallback (Gemini failed).")
    return _fallback_prescription(disease_text, context)