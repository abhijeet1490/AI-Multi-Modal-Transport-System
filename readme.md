You are an expert Full Stack Developer and Data Scientist building a university final year project titled "AI-Powered Multi-Modal Transport System".

Project Goal: Build a functional prototype that allows a user to enter shipment details (Origin, Destination, Weight, Distance) and receive an AI-optimized route recommendation with three key metrics: Cost, Time, and CO2 Emissions.

The Tech Stack (Strict Constraints):

AI Engine: Python + Scikit-learn. Use RandomForestRegressor wrapped in MultiOutputRegressor. You must generate synthetic training data since we do not have a real dataset.

Backend: Python + Flask. Lightweight API. Must handle CORS. Must fail gracefully (use mock data) if the AI model file is missing.

Frontend: Single index.html file. Use Tailwind CSS (via CDN) for styling. Use Vanilla JavaScript (fetch) to talk to the backend. No React/Vue/Node.js build steps.

Design Philosophy (Vibe):

Aesthetic: Clean, professional "Logistics Dashboard". Colors: Slate Blue, White, Emerald Green for sustainability.

Code Style: Minimal, commented, and robust.

Architecture:

train_model.py: Generates data, trains model, saves model.pkl.

app.py: Flask server, loads model.pkl, exposes /predict_route.

index.html: The UI.

Current Status: I am starting from scratch. I need you to guide me file-by-file. Acknowledge this context and wait for my first file request.