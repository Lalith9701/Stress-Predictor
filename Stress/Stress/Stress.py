import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import random
import sys
import os
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image, ImageTk

# Get the correct path to external files when running the executable
if getattr(sys, 'frozen', False):  # If running as a frozen executable (pyinstaller)
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# Paths
model_path = os.path.join(base_path, 'stress_cat_model.pkl')
num_model_path = os.path.join(base_path, 'stress_num_model.pkl')
scaler_path = os.path.join(base_path, 'scaler.pkl')
file_path = os.path.join(base_path, 'Final_Stress_Dataset.csv')
bg_image_path = os.path.join(base_path, 'stress.jpg')  # optional

# Try loading models/data with helpful error messages
try:
    cat_model = joblib.load(model_path)
except Exception as e:
    print(f"Could not load categorical model at {model_path}: {e}")
    cat_model = None

try:
    num_model = joblib.load(num_model_path)
except Exception as e:
    print(f"Could not load numerical model at {num_model_path}: {e}")
    num_model = None

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Could not load scaler at {scaler_path}: {e}")
    scaler = None

try:
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    data.columns = data.columns.str.strip()
except Exception as e:
    print(f"Could not load dataset at {file_path}: {e}")
    data = None

# If data is present, prepare encoders and columns (fallback safe defaults if not)
categorical_columns = ['Gender', 'Occupation', 'Traffic_Peak_Time', 'Road_Condition', 'Climate_condition',
                       'Medical_Problems', 'Suggestions', 'Martial_Status', 'BP', 'Traffic_congestion', 'Stressed',
                       'Late_to_work', 'State', 'Area', 'District', 'City', 'reason_of_traffic', 'Transport_Mode',
                       'Dietary_Habits', 'Alcohol_Smoking_Drugs', 'Exposure_to_Stressful_Events', 'Other']
numeric_columns = ['Age', 'Height(ft)', 'Weight(Kg)', 'HeartRate(bpm)', 'Priority', 'Traffic_Struck',
                   'Problem_by_stress', 'Work_Stress', 'Rush_hour', 'Accident', 'Experience', 'Work_Distance',
                   'Navigation_usage', 'Medical_History', 'Affect_Mood', 'Route_Change', 'Road_rage',
                   'Employer_support', 'Remote_work', 'Stuck_in_Traffic', 'Job_stress', 'sweating_issues',
                   'Traffic _Management', 'Work_Load', 'Sleep_Patterns', 'Exercise_Frequency', 'Exercise_Duration',
                   'Caffeine_Intake']

label_encoders = {}
label_encoders_target = {}

if data is not None:
    # Encode categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        # Ensure all values are strings to avoid unexpected types
        data[col] = data[col].astype(str)
        try:
            data[col] = le.fit_transform(data[col])
        except Exception:
            # If something fails, still store the encoder with single class to avoid KeyError later
            le.fit(['Unknown'])
            data[col] = le.transform(data[col].fillna('Unknown').astype(str))
        label_encoders[col] = le

    # Fill numeric missing values
    data.fillna(data.median(numeric_only=True), inplace=True)

    categorical_target_cols = ['Response_in_traffic', 'Stress_Impact']
    numerical_target_cols = ['Stress_Level', 'Anxiety_Levels', 'Depression_Levels']

    # Prepare target encoders
    for col in categorical_target_cols:
        le = LabelEncoder()
        # defensive: cast to str
        if col in data.columns:
            le.fit(data[col].astype(str))
        else:
            le.fit(['Unknown'])
        label_encoders_target[col] = le

else:
    # If data not loaded, create simple placeholder encoders with a safe default value
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(['Unknown'])
        label_encoders[col] = le
    for col in ['Response_in_traffic', 'Stress_Impact']:
        le = LabelEncoder()
        le.fit(['Less', 'Mild', 'Severe'])
        label_encoders_target[col] = le

# Prediction Function
def predict_stress():
    try:
        # collect inputs
        input_rows = []
        for col in categorical_columns:
            val = inputs[col].get().strip()
            if val == "":
                # default to first class if empty
                classes = list(label_encoders[col].classes_)
                if len(classes) > 0:
                    val = classes[0]
                else:
                    val = 'Unknown'
            # if value unseen, LabelEncoder.transform will raise -> map to 'Unknown' if present
            try:
                enc = label_encoders[col].transform([val])[0]
            except Exception:
                # try to map unseen to first class or to 0
                try:
                    enc = 0
                except Exception:
                    enc = 0
            input_rows.append(enc)

        for col in numeric_columns:
            raw = inputs[col].get().strip()
            if raw == "":
                # If empty numeric, default to 0
                num = 0.0
            else:
                try:
                    num = float(raw)
                except ValueError:
                    # invalid numeric input
                    messagebox.showerror("Input error", f"Numeric field '{col}' expects a number. Got '{raw}'.")
                    return
            input_rows.append(num)

        input_arr = np.array([input_rows])

        if scaler is not None:
            input_arr_scaled = scaler.transform(input_arr)
        else:
            input_arr_scaled = input_arr  # fallback

        # Categorical predictions
        if cat_model is not None:
            cat_preds = cat_model.predict(input_arr_scaled)  # expected shape (1, n_cat_targets) or (1,) depending model
        else:
            cat_preds = np.array([[0, 0]])  # fallback numeric-encoded labels

        # Numerical predictions
        if num_model is not None:
            num_preds = num_model.predict(input_arr_scaled)  # expected shape (1, 3)
        else:
            num_preds = np.array([[5.0, 5.0, 5.0]])

        # Decode categorical predictions safely
        # if cat_preds is 1D, make it 2D
        cat_preds = np.array(cat_preds)
        if cat_preds.ndim == 1:
            # if only one target predicted, create list; but we expect two targets (Response_in_traffic, Stress_Impact)
            cat_preds = cat_preds.reshape(1, -1)

        # Response_in_traffic decode
        try:
            resp_encoded = int(cat_preds[0][0])
            Response_in_Traffic = label_encoders_target['Response_in_traffic'].inverse_transform([resp_encoded])[0]
        except Exception:
            Response_in_Traffic = label_encoders_target['Response_in_traffic'].classes_[0]

        # Stress_Impact decode
        try:
            stress_imp_encoded = int(cat_preds[0][1])
            Stress_Impact = label_encoders_target['Stress_Impact'].inverse_transform([stress_imp_encoded])[0]
        except Exception:
            Stress_Impact = label_encoders_target['Stress_Impact'].classes_[0]

        # Numerical outputs and clipping
        Stress_Level = float(num_preds[0][0])
        Anxiety_Levels = float(num_preds[0][1])
        Depression_Levels = float(num_preds[0][2])

        Stress_Level = float(np.clip(Stress_Level, 0, 10))
        Anxiety_Levels = float(np.clip(Anxiety_Levels, 0, 10))
        Depression_Levels = float(np.clip(Depression_Levels, 0, 10))

        # Suggestion based on stress level
        if Stress_Level >= 7:
            Stress_Impact = 'Severe'
            tip = ("Deep breaths—your safety matters more than the rush. "
                   "Consider grounding exercises, breathing techniques or professional help like CBT.")
        elif 4 <= Stress_Level < 7:
            Stress_Impact = 'Mild'
            tip = ("Take deep breaths, short walks, or breathing exercises. "
                   "Good sleep and routine help reduce mild stress.")
        else:
            Stress_Impact = 'Less'
            tip = ("Relax, listen to a podcast or music, and focus on the present moment.")

        # Show results (also print to console so VSCode terminal shows it)
        result_text = (
            f"Stress Level: {Stress_Level:.2f}\n"
            f"Anxiety Levels: {Anxiety_Levels:.2f}\n"
            f"Depression Levels: {Depression_Levels:.2f}\n"
            f"Behavior in traffic: {Response_in_Traffic}\n"
            f"Stress Impact: {Stress_Impact}\n"
            f"Suggestion: {tip}"
        )
        print(result_text)
        messagebox.showinfo("Prediction Results", result_text)

    except Exception as e:
        # print traceback for debugging in console + notify user
        tb = traceback.format_exc()
        print("Exception in predict_stress():\n", tb)
        messagebox.showerror("Error", f"An error occurred. See console for details.\n{str(e)}")

# Tkinter UI Setup
root = tk.Tk()
root.title("Stress Level Predictor")
root.geometry("1000x700")

# Background image (optional)
try:
    if os.path.exists(bg_image_path):
        img = Image.open(bg_image_path)
        try:
            resample = Image.Resampling.LANCZOS  # Pillow >=9.1
        except AttributeError:
            resample = Image.LANCZOS
        img = img.resize((1000, 700), resample)
        bg_img = ImageTk.PhotoImage(img)
        bg_label = tk.Label(root, image=bg_img)
        bg_label.place(relwidth=1, relheight=1)
except Exception as e:
    print("Background image not loaded:", e)

frame = ttk.Frame(root, padding=8)
frame.place(relx=0.02, rely=0.02, relwidth=0.96, relheight=0.9)

inputs = {}

left_col = ttk.Frame(frame)
left_col.place(relx=0.01, rely=0.01, relwidth=0.48, relheight=0.96)
right_col = ttk.Frame(frame)
right_col.place(relx=0.51, rely=0.01, relwidth=0.48, relheight=0.96)

left_row = 0
right_row = 0

# Populate UI fields: use left/right alternation
for i, col in enumerate(categorical_columns + numeric_columns):
    target = left_col if i % 2 == 0 else right_col
    if target is left_col:
        row = left_row
    else:
        row = right_row

    lbl = ttk.Label(target, text=col)
    lbl.grid(row=row, column=0, sticky=tk.W, padx=4, pady=4)

    if col in categorical_columns:
        # use encoder classes if available
        classes = list(label_encoders[col].classes_) if col in label_encoders else ['Unknown']
        cmb = ttk.Combobox(target, values=classes)
        if len(classes) > 0:
            cmb.set(classes[0])
        cmb.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=4, pady=4)
        inputs[col] = cmb
    else:
        ent = ttk.Entry(target)
        ent.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=4, pady=4)
        inputs[col] = ent

    if target is left_col:
        left_row += 1
    else:
        right_row += 1

btn = ttk.Button(frame, text="Predict Stress Levels", command=predict_stress)
btn.place(relx=0.4, rely=0.96)

root.mainloop()
