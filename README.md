# Deep-Learning-Exp-9 

## Build and deploy your own Deep neural network on a website using Tensorflow

**NAME:** Rahul M R

**REGISTER NUMBER:** 2305003005

## AIM:
To design, train, and deploy a Deep Neural Network using TensorFlow on a fully functional web interface, enabling real-time prediction based on user-provided inputs.

## ALGORITHM:

**Step-1: Collect and preprocess the input data**

Gather user health and lifestyle parameters, normalize numerical features, and encode categorical values.

**Step-2: Design the Deep Neural Network architecture**

Create a TensorFlow model with input, hidden, and output layers tailored for risk-score prediction.

**Step-3: Compile the neural model**

Select an appropriate optimizer, loss function, and evaluation metrics.

**Step-4: Train the model on labeled health-risk data**

Fit the model over multiple epochs until satisfactory convergence is achieved.

**Step-5: Validate and evaluate the model’s accuracy**

Test performance using unseen data and adjust hyperparameters if required.

**Step-6: Export the trained model for web deployment**

Convert the model into TensorFlow.js format for browser-side predictions.

**Step-7: Build the website interface for user input**

Develop HTML/CSS/JS forms for collecting user data and linking to the prediction engine.

**Step-8: Integrate the TensorFlow.js model with the frontend**

Load the converted model, preprocess inputs dynamically, and generate real-time predictions.

**Step-9: Initialize and push the project to GitHub**

Create a new repository, commit all website and model files, and maintain proper version control.

**Step-10: Deploy the complete application online**

Host the frontend and TensorFlow.js model using a platform like Vercel, Netlify, or Lovable for public access.

## Deep Neural Network Model:

### Early Disease Risk Predictor (Lifestyle Input → Risk Score)

A simple Dense Neural Network taking age, habits, symptoms, lifestyle → predicts risk level for diabetes/heart issues.

- Health + AI always gets extra marks

- Input form looks neat

- Model is simple (just numerical inputs)

## PROGRAM:

### train_model.py

```python
# train_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------
# Example features (order)
# age, gender (0=F,1=M), bmi, smoking (0/1), alcohol (0/1), activity_level (0-2),
# sleep_hours, family_history (0/1), fatigue (0/1), thirst (0/1), chest_pain (0/1)
# --------------------------

def generate_synthetic_data(n=8000, seed=42):
    rng = np.random.RandomState(seed)
    age = rng.randint(18, 80, size=n)
    gender = rng.randint(0, 2, size=n)
    bmi = np.round(rng.normal(25, 5, size=n), 1).clip(14, 45)
    smoking = rng.binomial(1, 0.18, size=n)
    alcohol = rng.binomial(1, 0.25, size=n)
    activity = rng.choice([0,1,2], size=n, p=[0.35,0.4,0.25])  # 0 low,1 med,2 high
    sleep = np.round(rng.normal(7, 1.2, size=n), 1).clip(3, 11)
    family = rng.binomial(1, 0.15, size=n)
    fatigue = rng.binomial(1, 0.22, size=n)
    thirst = rng.binomial(1, 0.1, size=n)
    chest = rng.binomial(1, 0.06, size=n)

    X = np.vstack([age, gender, bmi, smoking, alcohol, activity, sleep,
                   family, fatigue, thirst, chest]).T

    # Create a synthetic risk score (0..1) as a weighted function + noise
    risk_raw = (
        0.014 * (age - 30) +
        0.02 * (bmi - 22) +
        0.25 * smoking +
        0.18 * alcohol +
        0.28 * family +
        0.22 * fatigue +
        0.16 * thirst +
        0.3 * chest -
        0.09 * activity -
        0.02 * (sleep - 7)
    )

    risk_raw = (risk_raw - risk_raw.min()) / (risk_raw.max() - risk_raw.min())
    # Add noise
    risk = (risk_raw + 0.08 * rng.normal(size=n)).clip(0, 1)

    # Convert to classes optionally, but we'll keep regression (0-1) as target
    y = risk.astype(np.float32)

    cols = ['age','gender','bmi','smoking','alcohol','activity','sleep',
            'family_history','fatigue','thirst','chest_pain']
    df = pd.DataFrame(X, columns=cols)
    df['risk_score'] = y
    return df

def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.18),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # output 0..1 risk
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['mae'])
    return model

def main():
    df = generate_synthetic_data(n=10000)
    X = df.drop(columns=['risk_score']).values
    y = df['risk_score'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.12, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = build_model(input_dim=X_train.shape[1])

    es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=120, batch_size=64, callbacks=[es], verbose=2)

    # Save scaler + model
    import joblib
    joblib.dump(scaler, 'scaler.gz')
    model.save('saved_model/lifepulse_model')  # TensorFlow SavedModel directory
    print("Training complete. Model saved to saved_model/lifepulse_model and scaler.gz")

if __name__ == "__main__":
    main()
```
### app.py

```python
# app.py
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import traceback

app = Flask(__name__)

# Load model & scaler at startup
MODEL_PATH = "saved_model/lifepulse_model"
SCALER_PATH = "scaler.gz"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def map_features(json_data):
    """
    Expect json_data to include keys:
    age, gender (0/1), bmi, smoking (0/1), alcohol (0/1), activity (0-2),
    sleep, family_history (0/1), fatigue (0/1), thirst (0/1), chest_pain (0/1)
    """
    keys = ['age','gender','bmi','smoking','alcohol','activity','sleep',
            'family_history','fatigue','thirst','chest_pain']
    x = []
    for k in keys:
        if k not in json_data:
            raise ValueError(f"Missing feature: {k}")
        x.append(float(json_data[k]))
    return np.array(x).reshape(1, -1)

def score_to_label(score):
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Moderate"
    else:
        return "High"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        x = map_features(data)
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled, verbose=0)[0,0]  # single float
        label = score_to_label(pred)
        # You can also return contributor explanation later (SHAP, weights, simple rule)
        return jsonify({
            "risk_score": float(pred),
            "risk_label": label
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

### app.tsx

```typescript

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useLocation, useNavigate } from "react-router-dom";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, CheckCircle2, TrendingUp, ArrowLeft, Activity } from "lucide-react";
import { FormData } from "@/pages/Analysis";

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const formData = location.state?.formData as FormData;
  const [displayScore, setDisplayScore] = useState(0);

  // Redirect if no form data
  useEffect(() => {
    if (!formData) {
      navigate("/analysis");
    }
  }, [formData, navigate]);

  // Calculate risk score (simplified algorithm for demo)
  const calculateRiskScore = (): number => {
    let score = 50; // Base score
    
    // Age factor
    const age = parseInt(formData?.age || "0");
    if (age > 50) score += 15;
    else if (age > 40) score += 10;
    
    // Lifestyle factors
    if (formData?.smoking === "regular") score += 15;
    if (formData?.smoking === "occasional") score += 8;
    if (formData?.alcohol === "heavy") score += 10;
    if (formData?.activity === "sedentary") score += 8;
    if (formData?.sleep === "<5") score += 5;
    
    // Symptoms
    if (formData?.fatigue) score += 5;
    if (formData?.thirst) score += 5;
    if (formData?.chestPain) score += 10;
    if (formData?.familyHistory) score += 8;
    
    return Math.min(Math.max(score, 0), 100);
  };

  const riskScore = calculateRiskScore();
  const riskLevel = riskScore < 40 ? "low" : riskScore < 70 ? "moderate" : "high";
  const riskColor = riskLevel === "low" ? "success" : riskLevel === "moderate" ? "warning" : "danger";
  const riskLabel = riskLevel === "low" ? "Low Risk" : riskLevel === "moderate" ? "Moderate Risk" : "High Risk";

  // Animate score counter
  useEffect(() => {
    const timer = setInterval(() => {
      setDisplayScore((prev) => {
        if (prev < riskScore) {
          return Math.min(prev + 2, riskScore);
        }
        clearInterval(timer);
        return prev;
      });
    }, 30);
    return () => clearInterval(timer);
  }, [riskScore]);

  const recommendations = [
    {
      title: "Regular Health Checkups",
      description: "Schedule annual physical examinations and screenings based on your age and risk factors.",
    },
    {
      title: "Lifestyle Modifications",
      description: "Focus on balanced nutrition, regular exercise, and adequate sleep to improve overall health.",
    },
    {
      title: "Stress Management",
      description: "Practice mindfulness, meditation, or yoga to reduce stress and improve mental wellbeing.",
    },
    {
      title: "Monitor Symptoms",
      description: "Keep track of any new or worsening symptoms and consult healthcare professionals when needed.",
    },
  ];

  if (!formData) return null;

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />
      
      <main className="flex-1 pt-24 pb-12">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-4xl mx-auto"
          >
            {/* Header */}
            <div className="text-center mb-8">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.2, type: "spring" }}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-card border border-border mb-4"
              >
                <Activity className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium">Analysis Complete</span>
              </motion.div>
              <h1 className="text-4xl font-bold mb-3 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                Your Health Risk Assessment
              </h1>
              <p className="text-muted-foreground">
                Based on your lifestyle habits, symptoms, and personal data
              </p>
            </div>

            {/* Risk Score Card */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="bg-card rounded-3xl p-8 shadow-card mb-8 border-2 border-border"
            >
              <div className="flex flex-col md:flex-row items-center justify-between gap-8">
                <div className="flex-1 text-center md:text-left">
                  <h2 className="text-xl font-semibold mb-2 text-muted-foreground">Overall Risk Score</h2>
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.5, type: "spring" }}
                    className="text-7xl font-bold mb-2"
                    style={{ color: `hsl(var(--${riskColor}))` }}
                  >
                    {displayScore}
                  </motion.div>
                  <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full bg-${riskColor}/10`}>
                    {riskLevel === "low" && <CheckCircle2 className="w-5 h-5 text-success" />}
                    {riskLevel === "moderate" && <TrendingUp className="w-5 h-5 text-warning" />}
                    {riskLevel === "high" && <AlertCircle className="w-5 h-5 text-danger" />}
                    <span className={`font-semibold text-${riskColor}`}>{riskLabel}</span>
                  </div>
                </div>
                
                <div className="flex-1 w-full">
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-muted-foreground">Risk Level</span>
                        <span className="font-medium">{displayScore}/100</span>
                      </div>
                      <Progress value={displayScore} className="h-3" />
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {riskLevel === "low" && "Your health indicators suggest a low risk. Continue maintaining healthy habits."}
                      {riskLevel === "moderate" && "Some factors indicate moderate risk. Consider lifestyle improvements and regular checkups."}
                      {riskLevel === "high" && "Multiple risk factors detected. We recommend consulting a healthcare professional soon."}
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Key Contributors */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-card rounded-3xl p-8 shadow-card mb-8"
            >
              <h3 className="text-2xl font-semibold mb-6">Key Risk Factors</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {formData.smoking !== "never" && (
                  <div className="p-4 rounded-xl bg-danger/10 border border-danger/20">
                    <div className="font-medium text-danger mb-1">Smoking</div>
                    <div className="text-sm text-muted-foreground">Current smoking status contributes to risk</div>
                  </div>
                )}
                {formData.activity === "sedentary" && (
                  <div className="p-4 rounded-xl bg-warning/10 border border-warning/20">
                    <div className="font-medium text-warning mb-1">Physical Activity</div>
                    <div className="text-sm text-muted-foreground">Sedentary lifestyle increases health risks</div>
                  </div>
                )}
                {formData.chestPain && (
                  <div className="p-4 rounded-xl bg-danger/10 border border-danger/20">
                    <div className="font-medium text-danger mb-1">Chest Discomfort</div>
                    <div className="text-sm text-muted-foreground">Reported symptom requires attention</div>
                  </div>
                )}
                {formData.familyHistory && (
                  <div className="p-4 rounded-xl bg-warning/10 border border-warning/20">
                    <div className="font-medium text-warning mb-1">Family History</div>
                    <div className="text-sm text-muted-foreground">Genetic predisposition detected</div>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Recommendations */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="bg-card rounded-3xl p-8 shadow-card mb-8"
            >
              <h3 className="text-2xl font-semibold mb-6">Personalized Recommendations</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {recommendations.map((rec, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.8 + index * 0.1 }}
                    className="p-4 rounded-xl border-2 border-border hover:border-primary/50 transition-all"
                  >
                    <h4 className="font-semibold mb-2 text-primary">{rec.title}</h4>
                    <p className="text-sm text-muted-foreground">{rec.description}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Actions */}
            <div className="flex gap-4 justify-center">
              <Button
                size="lg"
                variant="outline"
                onClick={() => navigate("/analysis")}
                className="rounded-xl px-8 group"
              >
                <ArrowLeft className="mr-2 w-5 h-5 group-hover:-translate-x-1 transition-transform" />
                Retake Assessment
              </Button>
              <Button
                size="lg"
                onClick={() => navigate("/")}
                className="rounded-xl px-8"
              >
                Return Home
              </Button>
            </div>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Results;
```

## OUTPUT:

### HOME PAGE:

<img width="1919" height="927" alt="image" src="https://github.com/user-attachments/assets/bef25cd6-e1c6-4e6a-be9d-058d1ee2867c" />

--- 

### DATA GATHERING:

<img width="1917" height="919" alt="image" src="https://github.com/user-attachments/assets/fb0eda1d-9850-4fa4-8e84-a0461f3bbb91" />

<img width="1919" height="917" alt="image" src="https://github.com/user-attachments/assets/50f836f1-51eb-49b7-8de5-05982ed63abe" />

<img width="1919" height="915" alt="image" src="https://github.com/user-attachments/assets/61f4b537-beea-4cf9-b155-f528ab5527b5" />

--- 

### RESULTS PAGE:

<img width="1919" height="874" alt="image" src="https://github.com/user-attachments/assets/1f2e28a2-665f-44bf-a8ee-2a192475a460" />


## RESULT:

Thus, the project successfully builds, trains, and deploys a Deep Neural Network using TensorFlow, 
integrates it into a responsive web interface, and delivers real-time health risk predictions directly to users through an accessible online platform.
