Write-Host "🐶 Starting CanineCare-AI-Vet setup..."

# 1. Go to project folder
Set-Location "CanineCare-AI-Vet-for-Dog-Breeds"

# 2. Create virtual environment if not exists
if (-Not (Test-Path "venv")) {
    Write-Host "⚙ Creating virtual environment..."
    python -m venv venv
}

# 3. Activate venv
Write-Host "✅ Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

# 4. Upgrade pip tools
Write-Host "⬆ Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

# 5. Define requirements
$requirements = @(
    "Flask==2.3.3",
    "tensorflow==2.15.0",
    "tensorflow-hub==0.16.1",
    "scikit-learn==1.3.2",
    "joblib==1.3.2",
    "opencv-python==4.9.0.80",
    "numpy==1.26.4",
    "pandas==2.2.2"
)

# 6. Install only missing packages
foreach ($pkg in $requirements) {
    $parts = $pkg -split "=="
    $name = $parts[0]
    $version = $parts[1]

    $installed = pip show $name 2>$null
    if ($installed -and ($installed -match "Version: $version")) {
        Write-Host "✔ $pkg already installed, skipping..."
    } else {
        Write-Host "📦 Installing $pkg ..."
        pip install $pkg
    }
}

# 7. Check required files
if (-Not (Test-Path "model/20220804-16551659632113-all-images-Adam.h5")) {
    Write-Host "❌ Missing: model/20220804-16551659632113-all-images-Adam.h5"
    exit 1
}
if (-Not (Test-Path "model/dogModel1.pkl")) {
    Write-Host "❌ Missing: model/dogModel1.pkl"
    exit 1
}
if (-Not (Test-Path "data/dog_data_09032022.csv")) {
    Write-Host "❌ Missing: data/dog_data_09032022.csv"
    exit 1
}

# 8. Run Flask app
Write-Host "🚀 Starting Flask app..."
$env:FLASK_APP="app.py"
$env:FLASK_ENV="development"
python app.py