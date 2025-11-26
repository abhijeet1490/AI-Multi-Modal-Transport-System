# AI-Powered Multi-Modal Transport System

## 📋 Project Overview

This is a university final year project prototype that demonstrates an **AI-powered logistics optimization system**. The application allows users to input shipment details (Origin, Destination, Weight, Distance, Traffic Level) and receive intelligent route recommendations with three key metrics:

- **💰 Total Cost** - Estimated freight cost in USD
- **⏱️ Time Estimate** - Approximate transit time in hours
- **🌱 CO₂ Emissions** - Carbon footprint in kilograms

The system uses a machine learning model (RandomForest) trained on synthetic logistics data to predict these metrics, with a graceful fallback to mock calculations if the model is unavailable.

---

## 🏗️ Architecture & Components

The project follows a **3-tier architecture**:

### 1. **AI Engine** (`train_model.py`)
- **Technology**: Python + Scikit-learn
- **Model**: RandomForestRegressor wrapped in MultiOutputRegressor
- **Purpose**: Generates synthetic training data and trains a multi-output regression model
- **Output**: Saves trained model as `logistics_model.pkl`

### 2. **Backend API** (`app.py`)
- **Technology**: Python + Flask
- **Purpose**: RESTful API server that loads the ML model and serves predictions
- **Features**:
  - CORS enabled for frontend communication
  - Graceful fallback to mock mode if model file is missing
  - Input validation and error handling
- **Port**: Runs on `http://127.0.0.1:5001` (configurable)

### 3. **Frontend** (`index.html`)
- **Technology**: Vanilla HTML/CSS/JavaScript
- **Styling**: Tailwind CSS (via CDN)
- **Purpose**: Single-page web interface for user interaction
- **Features**: Responsive design, loading states, error handling

---

## 🔄 How It Works (Project Flow)

### **Step 1: Model Training** (One-time setup)
```
train_model.py
    ↓
Generates 1,000 synthetic shipments
    ↓
Creates realistic formulas for Cost, Time, CO₂
    ↓
Trains RandomForest model (MultiOutputRegressor)
    ↓
Saves model to logistics_model.pkl
```

**Data Generation Logic:**
- **Input Features**: `distance_km`, `weight_kg`, `traffic_level` (1-10)
- **Target Formulas**:
  - **Cost** = Distance × 1.5 + Weight × 0.05 + Traffic × 8 + noise
  - **Time** = Distance / (70 - Traffic × 3) + noise
  - **CO₂** = Distance × 0.18 + Weight × 0.0004 + Traffic × 1.5 + noise

### **Step 2: Backend Server** (Runtime)
```
app.py starts
    ↓
Attempts to load logistics_model.pkl
    ↓
    ├─ Success → Model loaded, mock_mode = False
    └─ Failure → mock_mode = True (uses simple formulas)
    ↓
Flask server listens on port 5001
    ↓
Ready to accept POST requests at /predict_route
```

### **Step 3: User Interaction** (Runtime)
```
User opens index.html in browser
    ↓
Fills form: Origin, Destination, Weight, Distance, Traffic
    ↓
Clicks "Calculate Route"
    ↓
JavaScript sends POST request to /predict_route
    ↓
Backend processes request:
    ├─ If model available → Uses ML prediction
    └─ If mock_mode → Uses deterministic formulas
    ↓
Returns JSON: {success: true, data: {cost, time, co2}}
    ↓
Frontend displays results in 3 metric cards
```

---

## 🚀 Setup & Installation

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### **Step 1: Install Dependencies**

Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `Flask` - Web framework
- `flask-cors` - Cross-Origin Resource Sharing support
- `scikit-learn` - Machine learning library
- `pandas` - Data manipulation (used by scikit-learn)
- `numpy` - Numerical computing
- `pickle` - Built-in Python library (no installation needed)

### **Step 2: Train the Model**

Generate the machine learning model:

```bash
python train_model.py
```

**Expected Output:**
```
Generating synthetic data...
Training RandomForest multi-output model...
Saving model to 'logistics_model.pkl'...
Running a sample prediction...
Sample Prediction (single shipment):
  Input  -> distance_km=750, weight_kg=1200, traffic_level=5
  Output -> cost_usd=1,234.56
           time_hours=12.34
           co2_kg=145.67
Done.
```

This creates `logistics_model.pkl` in the project root directory.

**Note**: If you skip this step, the backend will automatically switch to `MOCK_MODE` and use simple formulas instead of ML predictions.

---

## ▶️ Running the Project

### **Step 1: Start the Backend Server**

```bash
python app.py
```

**Expected Output:**
```
[INFO] Loaded model from 'logistics_model.pkl'.
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5001
```

**Troubleshooting:**
- If port 5001 is in use, edit `app.py` line 231 and change `port=5001` to another port (e.g., `port=5002`)
- Update the `API_URL` in `index.html` line 209 to match the new port

### **Step 2: Open the Frontend**

Open `index.html` in your web browser:

**Option A: Direct File Open**
- Double-click `index.html` or right-click → "Open with" → Browser

**Option B: Simple HTTP Server** (Recommended for CORS)
```bash
# Python 3
python -m http.server 8000

# Then open: http://localhost:8000/index.html
```

### **Step 3: Test the Application**

1. Fill in the form:
   - **Origin**: e.g., "Mumbai, IN"
   - **Destination**: e.g., "Delhi, IN"
   - **Weight**: e.g., `1200` (kg)
   - **Distance**: e.g., `750` (km)
   - **Traffic Level**: e.g., `5` (1-10 scale)

2. Click **"Calculate Route"**

3. View results in the **Results Dashboard**:
   - Total Cost (dark card)
   - Time Estimate (blue card)
   - CO₂ Emissions (green card)

---

## 📁 File Structure

```
Project Prot/
│
├── train_model.py          # AI model training script
├── app.py                  # Flask backend server
├── index.html              # Frontend web interface
├── requirements.txt        # Python dependencies
├── logistics_model.pkl     # Trained ML model (generated)
├── readme.md              # Original project brief
└── READ2ME.md             # This documentation file
```

---

## 🔌 API Endpoints

### **GET /health**
Check server status and model availability.

**Response:**
```json
{
  "status": "ok",
  "mock_mode": false
}
```

**Example:**
```bash
curl http://127.0.0.1:5001/health
```

---

### **POST /predict_route**
Get route predictions for a shipment.

**Request Body:**
```json
{
  "distance": 750.0,
  "weight": 1200.0,
  "traffic": 5
}
```

**Success Response:**
```json
{
  "success": true,
  "data": {
    "cost": 1234.56,
    "time": 12.34,
    "co2": 145.67
  },
  "mock_mode": false
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Missing fields: distance, weight"
}
```

**Example (using curl):**
```bash
curl -X POST http://127.0.0.1:5001/predict_route \
  -H "Content-Type: application/json" \
  -d '{"distance": 750, "weight": 1200, "traffic": 5}'
```

---

## 🛠️ Technologies Used

### **Backend**
- **Python 3.8+** - Programming language
- **Flask** - Lightweight web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Scikit-learn** - Machine learning library
  - `RandomForestRegressor` - Base regressor
  - `MultiOutputRegressor` - Multi-target wrapper
- **NumPy** - Numerical computing
- **Pickle** - Model serialization

### **Frontend**
- **HTML5** - Markup
- **Tailwind CSS** (CDN) - Utility-first CSS framework
- **Vanilla JavaScript** - No frameworks, pure JS
- **Fetch API** - HTTP requests

### **Design Philosophy**
- **Colors**: Slate Blue, White, Emerald Green (sustainability theme)
- **Font**: Inter (Google Fonts)
- **Style**: Clean, professional logistics dashboard
- **Responsive**: Mobile-first, stacks on small screens

---

## 🧪 Testing the System

### **Test Model Training**
```bash
python train_model.py
# Verify: logistics_model.pkl is created
```

### **Test Backend API**
```bash
# Start server
python app.py

# In another terminal, test health endpoint
curl http://127.0.0.1:5001/health

# Test prediction endpoint
curl -X POST http://127.0.0.1:5001/predict_route \
  -H "Content-Type: application/json" \
  -d '{"distance": 500, "weight": 800, "traffic": 3}'
```

### **Test Mock Mode**
```bash
# Temporarily rename model file
mv logistics_model.pkl logistics_model.pkl.backup

# Restart server - should show MOCK_MODE warning
python app.py

# Test - should still return predictions (using formulas)
curl -X POST http://127.0.0.1:5001/predict_route \
  -H "Content-Type: application/json" \
  -d '{"distance": 500, "weight": 800, "traffic": 3}'

# Restore model
mv logistics_model.pkl.backup logistics_model.pkl
```

---

## 🐛 Troubleshooting

### **Port Already in Use**
**Error**: `Address already in use` or `Port 5001 is in use`

**Solution**:
1. Change port in `app.py` line 231: `port=5002`
2. Update `API_URL` in `index.html` line 209: `http://127.0.0.1:5002/predict_route`

### **CORS Errors in Browser**
**Error**: `CORS policy: No 'Access-Control-Allow-Origin' header`

**Solution**:
- Ensure Flask-CORS is installed: `pip install flask-cors`
- Verify `CORS(app)` is called in `app.py` line 110
- Use a local HTTP server instead of opening `index.html` directly

### **Model Not Found**
**Error**: `[WARN] Model file 'logistics_model.pkl' not found`

**Solution**:
- Run `python train_model.py` to generate the model
- Or the system will automatically use `MOCK_MODE` (still functional)

### **Import Errors**
**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

---

## 📊 Model Details

### **Training Configuration**
- **Samples**: 1,000 synthetic shipments
- **Features**: 3 (distance_km, weight_kg, traffic_level)
- **Targets**: 3 (cost_usd, time_hours, co2_kg)
- **Algorithm**: RandomForestRegressor
- **Estimators**: 200 trees
- **Random Seed**: 42 (for reproducibility)

### **Feature Ranges**
- **Distance**: 10 - 2,000 km
- **Weight**: 10 - 5,000 kg
- **Traffic**: 1 - 10 (integer scale)

### **Prediction Accuracy**
The model is trained on synthetic data with realistic formulas and noise. In a production system, you would train on real historical logistics data for better accuracy.

---

## 🎯 Future Enhancements

Potential improvements for a production system:

1. **Real Dataset**: Replace synthetic data with actual logistics records
2. **Multi-Modal Routes**: Add support for different transport modes (truck, rail, air, sea)
3. **Route Visualization**: Display route on an interactive map
4. **Historical Data**: Store predictions and compare with actual results
5. **User Authentication**: Add login/signup for personalized dashboards
6. **Database Integration**: Store shipments, routes, and predictions
7. **Advanced ML**: Experiment with neural networks or ensemble methods
8. **Real-time Traffic**: Integrate live traffic APIs (Google Maps, HERE, etc.)

---

## 📝 Notes

- This is a **prototype** for demonstration purposes
- The model uses **synthetic data** - real-world accuracy would require actual logistics datasets
- The frontend is a **single HTML file** - no build process required
- The backend is designed for **development** - use a production WSGI server (Gunicorn, uWSGI) for deployment
- All code is **commented** and follows clean coding practices

---

## 👤 Author

University Final Year Project - AI-Powered Multi-Modal Transport System

---

## 📄 License

This project is created for educational purposes as part of a university final year project.

---

**Last Updated**: 2024

**Version**: 1.0.0 (Prototype)

