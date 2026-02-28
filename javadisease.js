// static/js/javadisease.js

let reference4 = null;

// 1) SYMPTOM-BASED DISEASE PREDICTION
async function predictDisease() {
  // collect selected symptoms
  const symptoms = [];
  for (let i = 1; i <= 5; i++) {
    const sel = document.getElementById(`symptom${i}`);
    if (sel && sel.value) symptoms.push(sel.value);
  }
  if (!symptoms.length) {
    alert("⚠️ Please select at least one symptom.");
    return;
  }

  // show loading note
  document.getElementById("loaderNote").textContent = "⏳ Predicting disease…";

  try {
    const res = await fetch("/predict_disease", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symptoms })
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const data = await res.json();
    document.getElementById("loaderNote").textContent = "";

    // you can display however you like; here, a simple alert:
    alert(`Possible disease: ${data.disease}\nMore info: ${data.url}`);
  } catch (err) {
    document.getElementById("loaderNote").textContent = "";
    console.error(err);
    alert("Error predicting disease: " + err.message);
  }
}

// 2) SHOW/HIDE IMAGE-UPLOAD UI
function showImagePredict() {
  document.getElementById("imagePredictSection").style.display = "block";
}

// 3) GEMINI SKIN-DISEASE PREDICTION
async function predictSkinDisease() {
  const fileInput = document.getElementById("skinImage");
  const resultDiv = document.getElementById("skinResult");
  if (!fileInput.files.length) {
    alert("⚠️ Please choose an image first.");
    return;
  }

  resultDiv.textContent = "⏳ Predicting skin disease…";

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  try {
    const res = await fetch("/predict_skin_gemini", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const data = await res.json();
    if (data.error) {
      resultDiv.textContent = "❌ " + data.error;
    } else {
      resultDiv.textContent = "✅ Prediction: " + data.result;
    }
  } catch (err) {
    console.error(err);
    resultDiv.textContent = "⚠️ Error: " + err.message;
  }
}

// 4) BACK BUTTON
function goBack() {
  const urlParams = new URLSearchParams(window.location.search);
  const animal = urlParams.get("animal") || "";
  window.location.href = `/two?animal=${encodeURIComponent(animal)}`;
}

// 5) INITIALIZATION (populate symptoms dropdowns)
async function init() {
  const urlParams = new URLSearchParams(window.location.search);
  const reference1 = urlParams.get("animal");
  const reference2 = urlParams.get("breed");

  document.getElementById("animalDisplay").textContent = reference1 || "Not selected";
  document.getElementById("breedDisplay").textContent = reference2 || "Not selected";

  if (reference2) {
    try {
      const res = await fetch("/static/data/jesus.json");
      if (!res.ok) throw new Error(`Failed to load symptoms JSON (${res.status})`);
      const data = await res.json();
      const breedData = data[reference2];
      if (breedData) {
        for (let i = 1; i <= 5; i++) {
          const sel = document.getElementById(`symptom${i}`);
          sel.innerHTML = `<option value="">Select Symptom ${i}</option>`;
          breedData.forEach(sym => {
            const opt = document.createElement("option");
            opt.value = sym;
            opt.textContent = sym;
            sel.appendChild(opt);
          });
        }
      }
    } catch (err) {
      console.error("Error loading symptoms:", err);
    }
  }
}

document.addEventListener("DOMContentLoaded", init);