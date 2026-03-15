# 🌾 Agricultural Water Needs Prediction API

## What's in this folder

```
flask_api/
├── app.py              ← Flask API server (main code)
├── requirements.txt    ← Python dependencies
├── Procfile            ← Start command for Railway/Render
├── Dockerfile          ← Container config (optional)
├── .gitignore          ← Git ignore rules
└── README.md           ← You are here
```

## You also need these files from Colab (Step 2)

After running the Colab notebook, unzip `water_prediction_model.zip` and copy these files into this same folder:

```
├── water_model.pkl         ← Trained ML model
├── encoder_crop.pkl        ← Crop label encoder
├── encoder_soil.pkl        ← Soil label encoder  
├── encoder_stage.pkl       ← Growth stage encoder
├── encoder_irrig.pkl       ← Irrigation method encoder
├── feature_cols.pkl        ← Feature column order
├── model_metadata.pkl      ← Model info
```

---

## STEP 3: Test the API Locally

### 3A — Set up Python environment

```bash
cd flask_api
python -m venv venv

# Activate:
# Windows:   venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### 3B — Start the API

```bash
python app.py
```

You should see:
```
Loading model and encoders...
✅ Model loaded: XGBRegressor (R²=0.9876)
 * Running on http://0.0.0.0:5000
```

### 3C — Test with curl (open another terminal)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "crop_type": "wheat",
    "growth_stage": "mid_season",
    "temp_max": 38.0,
    "temp_min": 22.0,
    "humidity": 35,
    "wind_speed": 3.2,
    "solar_radiation": 25.0,
    "rainfall": 0.0,
    "soil_type": "clay_loam",
    "soil_moisture": 28.0,
    "field_area_ha": 2.0,
    "irrigation_method": "drip"
  }'
```

### 3D — Other useful endpoints

```bash
# Check if API is alive
curl http://localhost:5000/health

# Get valid dropdown values (for frontend)
curl http://localhost:5000/info
```

---

## STEP 5: Deploy the API Online

### Option A: Railway (Recommended — Easiest)

1. Go to https://github.com → create a new repository called `water-api`
2. Push this entire folder to it:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/water-api.git
   git push -u origin main
   ```
3. Go to https://railway.app → Sign in with GitHub
4. Click **"New Project"** → **"Deploy from GitHub Repo"** → select `water-api`
5. Railway auto-detects Python + Procfile and deploys
6. Click **"Settings"** → **"Generate Domain"** to get your public URL
7. Your API is now live at: `https://water-api-production-XXXX.up.railway.app`

### Option B: Render (Also Free)

1. Push to GitHub (same as above)
2. Go to https://render.com → New → Web Service → Connect your repo
3. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Click Deploy
5. Your URL: `https://water-api-XXXX.onrender.com`

### Test your deployed API

```bash
curl https://YOUR-DEPLOYED-URL.com/health
curl https://YOUR-DEPLOYED-URL.com/info
```

---

## STEP 4: Build the Website with Lovable

See LOVABLE_PROMPT.md for the complete prompt to paste into Lovable.

After Lovable generates the app, change the API URL to your deployed Railway/Render URL.
