# generate_prescription_logic.py
"""
Robust prescription-generation wrapper that calls VertexAI / Gemini
to generate a JSON prescription. If the LLM response cannot be parsed
or validated, a deterministic fallback (rule-based) prescription is returned.

Expected input (data): dict possibly containing keys:
 - "disease" (preferred)
 - "symptoms" (list or comma string)
 - "breed", "age", "weight", "sex", "medical_history"
 - "petNo" (optional, used for PDF generation)

Returns a dict with keys:
 - "disease": string
 - "prescription_plan": list of dicts (date, time, medicine, dosage, route, duration, notes)
 - "diet_plan": list of dicts (date, feeding_time, food_type, quantity, notes)
 - "explanation": string
 - optionally "pdf_url" if pdf generation succeeds
"""

import os
import json
import traceback
from datetime import datetime

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

    for ent in obj['prescription_plan']:
        if not isinstance(ent, dict):
            return False, "prescription_plan entries must be objects"
    for ent in obj['diet_plan']:
        if not isinstance(ent, dict):
            return False, "diet_plan entries must be objects"

    return True, None


def _fallback_prescription(disease_keyword, context):
    d = (disease_keyword or "").lower()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    morning = "09:00"
    evening = "18:00"

    mapping = {
        "parvo": {
            "prescription_plan": [
                {"date": today, "time": morning, "medicine": "IV fluids (Ringer Lactate)", "dosage": "As per hydration status", "route": "injectable", "duration": "3 days", "notes": "Hospitalize if severe"},
                {"date": today, "time": evening, "medicine": "Metronidazole", "dosage": "20 mg/kg", "route": "oral", "duration": "5 days", "notes": "Give after food if tolerated"}
            ],
            "diet_plan": [
                {"date": today, "feeding_time": "morning", "food_type": "Boiled chicken broth", "quantity": "Small frequent portions", "notes": "Avoid fatty foods"}
            ],
            "explanation": "Probable parvovirus: treat with fluids, antiemetics, and supportive nutrition. Consult your veterinarian immediately."
        },
        "tick": {
            "prescription_plan": [
                {"date": today, "time": morning, "medicine": "Doxycycline", "dosage": "10 mg/kg", "route": "oral", "duration": "14 days", "notes": "Give with food"},
                {"date": today, "time": morning, "medicine": "Imidocarb (if indicated)", "dosage": "6 mg/kg", "route": "injectable", "duration": "Single dose", "notes": "Administered by vet"}
            ],
            "diet_plan": [
                {"date": today, "feeding_time": "morning", "food_type": "High protein, palatable diet", "quantity": "As required", "notes": "Ensure hydration"}
            ],
            "explanation": "Tick-borne disease suspected. Doxycycline is commonly used; definitive therapy depends on lab confirmation."
        },
        "allergy": {
            "prescription_plan": [
                {"date": today, "time": morning, "medicine": "Cetirizine (antihistamine)", "dosage": "0.5-1 mg/kg", "route": "oral", "duration": "7 days", "notes": "Adjust per vet"},
                {"date": today, "time": evening, "medicine": "Topical barrier ointment (if local lesions)", "dosage": "Apply thin layer", "route": "topical", "duration": "As needed", "notes": "Use sparingly"}
            ],
            "diet_plan": [
                {"date": today, "feeding_time": "morning", "food_type": "Hypoallergenic diet trial", "quantity": "As recommended", "notes": "Trial for 8-12 weeks if food allergy suspected"}
            ],
            "explanation": "Allergic dermatitis suspected; options include antihistamines, topical therapy, and dietary trials."
        }
    }

    for key, plan in mapping.items():
        if key in d:
            return {
                "disease": context.get("disease") or disease_keyword or key.title(),
                "prescription_plan": plan["prescription_plan"],
                "diet_plan": plan["diet_plan"],
                "explanation": plan["explanation"]
            }

    # Generic fallback
    return {
        "disease": context.get("disease") or (disease_keyword or "Unknown").title(),
        "prescription_plan": [
            {"date": today, "time": morning, "medicine": "Supportive care (fluids, rest)", "dosage": "As per vet", "route": "as advised", "duration": "Variable", "notes": "See a veterinarian for tailored therapy"}
        ],
        "diet_plan": [
            {"date": today, "feeding_time": "morning", "food_type": "Easily digestible food", "quantity": "Small frequent portions", "notes": "Encourage fluids"}
        ],
        "explanation": "No validated prescription could be generated automatically. This is a conservative supportive-care plan — please consult a veterinarian."
    }


def generate_prescription_logic(data):
    context = {
        "disease": (data.get("disease") or "").strip(),
        "symptoms": data.get("symptoms", []),
        "breed": data.get("breed", ""),
        "age": data.get("age", ""),
        "weight": data.get("weight", ""),
        "sex": data.get("sex", ""),
        "medical_history": data.get("medical_history", ""),
        "petNo": data.get("petNo")
    }

    disease_text = context["disease"]
    symptoms_text = ", ".join(context["symptoms"]) if isinstance(context["symptoms"], (list, tuple)) else str(context["symptoms"])

    prompt = f"""
You are a veterinary AI assistant. Use the data below to produce a biologically accurate JSON treatment plan.

Pet Profile:
- Breed: {context['breed'] or 'Unknown'}
- Age: {context['age'] or 'Unknown'}
- Weight: {context['weight'] or 'Unknown'}
- Sex: {context['sex'] or 'Unknown'}
- Symptoms: {symptoms_text or 'None provided'}
- Medical History: {context['medical_history'] or 'None'}

IMPORTANT: If a likely disease is already given use it to focus the prescription.
If no disease provided, infer the most likely disease from symptoms.

Return EXACTLY this JSON schema (no extra keys):
{{
  "disease": "string",
  "prescription_plan": [
    {{"date": "YYYY-MM-DD","time": "HH:MM","medicine": "real drug name","dosage": "e.g., 10 mg/kg","route": "oral | topical | injectable","duration": "e.g., 5 days","notes": "instructions"}}
  ],
  "diet_plan": [
    {{"date": "YYYY-MM-DD","feeding_time": "morning | evening","food_type": "e.g., kibble","quantity": "e.g., 100 g","notes": "notes"}}
  ],
  "explanation": "Owner-friendly rationale"
}}
Respond ONLY with that JSON object and nothing else.
"""
    if disease_text:
        prompt = f"Known disease: {disease_text}\n\n" + prompt

    model_output = None
    if GenerativeModel and vertexai:
        try:
            if hasattr(vertexai, "init"):
                project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT")
                location = os.environ.get("VERTEX_LOCATION") or os.environ.get("LOCATION")
                if project and location:
                    vertexai.init(project=project, location=location)
                else:
                    vertexai.init()
            model = GenerativeModel("gemini-1.5-flash", temperature=0.2, max_output_tokens=512)
            if hasattr(model, "generate_content"):
                response = model.generate_content(prompt)
                model_output = getattr(response, "text", getattr(response, "content", str(response)))
            elif hasattr(model, "generate"):
                response = model.generate(prompt, temperature=0.2, max_output_tokens=512)
                model_output = getattr(response, "text", getattr(response, "content", str(response)))
        except Exception as e:
            print("LLM generation failed:", e)
            model_output = None

    parsed = None
    if model_output:
        try:
            parsed = _safe_json_load(model_output)
            valid, reason = _validate_result_schema(parsed)
            if not valid:
                print("LLM output validation failed:", reason)
                parsed = None
        except Exception as e:
            print("Failed to parse/validate LLM output:", e)
            parsed = None

    if parsed:
        result = {
            "disease": parsed.get("disease", context["disease"] or "Unknown"),
            "prescription_plan": parsed.get("prescription_plan", []),
            "diet_plan": parsed.get("diet_plan", []),
            "explanation": parsed.get("explanation", "")
        }
    else:
        result = _fallback_prescription(context["disease"] or symptoms_text, context)

    pet_no = context.get("petNo")
    if pet_no and pdf_generate:
        try:
            pdf_url = pdf_generate(result, pet_no)
            result["pdf_url"] = pdf_url
        except Exception:
            print("PDF generation failed:", traceback.format_exc())

    return result
