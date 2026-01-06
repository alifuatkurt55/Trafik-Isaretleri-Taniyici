# ============================================================
# 1. SİSTEM VE GPU YAPILANDIRMASI
# ============================================================
import os
import sys
import json 

current_python_dir = os.path.dirname(sys.executable)
conda_env_path = current_python_dir
conda_lib_path = os.path.join(conda_env_path, "Library", "bin")

print("\n" + "="*50)
print(f"🔧 SİSTEM BAŞLATILIYOR...")

os.environ['PATH'] = conda_lib_path + os.pathsep + os.environ['PATH']
os.environ['PATH'] = os.path.join(conda_env_path, "Library", "bin") + os.pathsep + os.environ['PATH']

if hasattr(os, 'add_dll_directory') and os.path.exists(conda_lib_path):
    try:
        os.add_dll_directory(conda_lib_path)
    except Exception as e:
        pass

# =======================================================
# 2. IMPORTLAR
# =======================================================
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import cv2
import io
import base64
import threading
import numpy as np
import pandas as pd
import time
import gc
from flask import Flask, render_template, request, jsonify
from PIL import Image

# =======================================================
# 3. GPU AYARLARI
# =======================================================
print("DONANIM KONTROLÜ...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU BULUNDU: {len(gpus)} adet aktif.")
    except RuntimeError as e:
        print(f"❌ GPU Ayar Hatası: {e}")
else:
    print("⚠️ GPU BULUNAMADI! CPU Modu.")
print("="*50 + "\n")

# =======================================================
# 4. KERAS & SKLEARN IMPORTLARI
# =======================================================
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize 
from sklearn.model_selection import train_test_split

# =========================
# Flask Setup
# =========================
app = Flask(__name__)
UPLOAD_FOLDER = "dataset/Upload"
MODEL_PATH = "model/traffic_sign_cnn_deep.h5"
IMG_SIZE = (64, 64)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("model", exist_ok=True)

# =========================
# Globals
# =========================
model = None
progress = {"running": False, "percent": 0, "status": ""}
logs = []
stop_training = False
train_history = None
training_thread = None

full_test_progress = 0
full_test_results = {}
plot_lock = threading.Lock() 

# =========================
# Image preprocessing
# =========================
def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return img

# =========================
# Model Training
# =========================
def train_model():
    global model, progress, stop_training, train_history
    stop_training = False
    progress.update({"running": True, "percent": 0, "status": "Eğitim başlatıldı..."})
    
    TRAIN_DIR = "dataset/Train"
    IMG_SIZE = (30, 30)
    classes = 43
    data, labels = [], []

    # Veri Yükleme
    for i in range(classes):
        class_dir = os.path.join(TRAIN_DIR, str(i))
        if not os.path.exists(class_dir): continue
        for fname in os.listdir(class_dir):
            try:
                img_path = os.path.join(class_dir, fname)
                image = Image.open(img_path).resize(IMG_SIZE)
                data.append(np.array(image))
                labels.append(i)
            except: pass
    
    data = np.array(data, dtype='float32') / 255.0
    labels = np.array(labels)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    y_train = to_categorical(y_train, classes)
    y_val = to_categorical(y_val, classes)

    # Model Mimarisi
    model = models.Sequential([
        layers.Conv2D(32, (5,5), activation='relu', input_shape=(30,30,3)),
        layers.Conv2D(64, (5,5), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.15),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.20),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(classes, activation='softmax'),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    progress.update({"percent": 10, "status": "Eğitim başladı..."})

    # Eğitim
    train_history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=128,
        epochs=35,
        verbose=0,
    )

    model.save(MODEL_PATH)
    
    # --- EĞİTİM GEÇMİŞİNİ KAYDET (YENİ) ---
    try:
        history_path = os.path.join("training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            # Numpy float32 tiplerini python float'a çevir
            h = {k: [float(val) for val in v] for k, v in train_history.history.items()}
            json.dump(h, f)
        print(f"Eğitim geçmişi kaydedildi: {history_path}")
    except Exception as e:
        print(f"Eğitim geçmişi kaydetme hatası: {e}")
    # --------------------------------------

    progress.update({"percent": 100, "running": False, "status": "Tamamlandı"})

# =========================
# Label Overview
# =========================
classes = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)', 9:'No passing',
    10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
    16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians',
    28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow',
    31:'Wild animals crossing', 32:'End speed + passing limits',
    33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only',
    36:'Go straight or right', 37:'Go straight or left', 38:'Keep right',
    39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing',
    42:'End no passing veh > 3.5 tons'
}

# =========================
# CNN PREDICT
# =========================
def cnn_predict(img_or_path, true_label=None):
    if model is None: return {"error": "Model yüklenmedi."}
    input_shape = model.input_shape[1:3]

    if isinstance(img_or_path, np.ndarray):
        img = Image.fromarray(img_or_path.astype("uint8")).convert("RGB")
        cv_img = cv2.cvtColor(img_or_path, cv2.COLOR_RGB2BGR)
    else:
        img = Image.open(img_or_path).convert("RGB")
        cv_img = cv2.imread(img_or_path)
        if cv_img is None: cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    img_resized = img.resize(input_shape)
    img_arr = np.array(img_resized, dtype="float32") / 255.0
    img_input = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_input, verbose=0)[0]
    prediction = int(np.argmax(preds))
    confidence = float(np.max(preds))
    predicted_name = classes.get(prediction, "Unknown")
    true_name = classes.get(true_label, "Unknown") if true_label is not None else "Unknown"

    cv_test = cv2.resize(cv_img, input_shape)
    _, buf1 = cv2.imencode(".png", cv2.cvtColor(cv_test, cv2.COLOR_BGR2RGB))
    test_b64 = f"data:image/png;base64,{base64.b64encode(buf1).decode()}"
    
    meta_b64 = test_b64
    meta_path = f"dataset/Meta/{prediction}.png"
    if os.path.exists(meta_path):
        meta_img = cv2.imread(meta_path)
        if meta_img is not None:
            _, buf2 = cv2.imencode(".png", meta_img)
            meta_b64 = f"data:image/png;base64,{base64.b64encode(buf2).decode()}"

    return {
        "prediction": prediction, "prediction_name": predicted_name,
        "class_name": true_name, "confidence": confidence,
        "test_img": test_b64, "meta_img": meta_b64
    }

# =========================
# Flask Routes
# =========================
@app.route("/")
def index(): return render_template("index.html")

@app.route("/start_training")
def start_training_route():
    global training_thread
    if training_thread is not None and training_thread.is_alive():
        return jsonify({"error": "Devam ediyor."})
    
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()
    return jsonify({"status": "Başlatıldı."})

@app.route("/stop_training")
def stop_training_route():
    global stop_training; stop_training = True
    return jsonify({"status": "Durduruldu."})

@app.route("/progress")
def get_progress(): return jsonify(progress)

@app.route("/logs")
def get_logs(): return jsonify({"logs": logs})

@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, model.input_shape[1:3])
    first = cnn_predict(resized)
    return jsonify(cnn_predict(resized, true_label=first["prediction"]))

@app.route("/predict_random_test")
def predict_random_test():
    test_csv = pd.read_csv(os.path.join("", "dataset", "Test.csv"))
    row = test_csv.sample(1).iloc[0]
    return jsonify(cnn_predict(os.path.join("dataset", row["Path"]), int(row["ClassId"])))

@app.route("/predict_batch_test/<int:num_samples>")
def predict_batch_test(num_samples):
    test_csv = pd.read_csv("dataset/Test.csv")
    samples = test_csv.sample(min(num_samples, len(test_csv)))
    results = [cnn_predict(os.path.join("dataset", row["Path"]), int(row["ClassId"])) for _, row in samples.iterrows()]
    return jsonify(results)

@app.route("/confusion_samples/<int:num_samples>")
def confusion_samples(num_samples):
    test_csv = pd.read_csv("dataset/Test.csv")
    samples = test_csv.sample(min(num_samples, len(test_csv)))
    y_true, y_pred = [], []
    for _, row in samples.iterrows():
        res = cnn_predict(os.path.join("", "dataset", row["Path"]))
        y_true.append(int(row["ClassId"]))
        y_pred.append(res["prediction"])
    
    with plot_lock:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=False, cmap="Blues")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close('all')
        return jsonify({"img": base64.b64encode(buf.getvalue()).decode()})

# =========================================================================
# ROC SAMPLES
# =========================================================================
@app.route("/roc_samples/<int:num_samples>")
def roc_samples(num_samples):
    if model is None: return jsonify({"error": "Model yüklenmedi."})
    
    try:
        test_csv = pd.read_csv("dataset/Test.csv")
        samples = test_csv.sample(min(num_samples, len(test_csv)))
        
        y_true = []
        y_scores = []
        input_shape = model.input_shape[1:3]

        for _, row in samples.iterrows():
            try:
                img_path = os.path.join("dataset", row["Path"])
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, input_shape).astype("float32") / 255.0
                X = np.expand_dims(img, axis=0)
                
                preds = model.predict(X, verbose=0)[0]
                y_scores.append(preds)
                y_true.append(int(row["ClassId"]))
            except: continue

        with plot_lock:
            n_classes = 43
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            y_scores = np.array(y_scores)

            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-average ROC (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Eğrisi ({num_samples} Rastgele Örnek)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
            plt.close('all')
            return jsonify({"img": base64.b64encode(buf.getvalue()).decode()})

    except Exception as e:
        print(f"ROC Hatası: {e}")
        return jsonify({"error": str(e)})

# =========================================================================
# EPOCH PLOT (Öğrenme Eğrisi) - DOSYADAN OKUMA DESTEKLİ
# =========================================================================
@app.route("/epoch_plot")
def epoch_plot():
    global train_history
    data = None

    # 1. Bellekten oku
    if train_history is not None:
        data = train_history.history
    # 2. Dosyadan oku
    else:
        history_path = os.path.join("training_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Dosya okuma hatası: {e}")

    if data is None: 
        return jsonify({"error": "Henüz eğitim verisi yok."})
    
    with plot_lock:
        plt.figure(figsize=(10,5))
        if "accuracy" in data:
            plt.plot(data["accuracy"], label="Training Accuracy", linewidth=2)
        if "loss" in data:
            plt.plot(data["loss"], label="Training Loss", linewidth=2, linestyle="--")
            
        plt.title("Öğrenme Eğrisi (Learning Curve)")
        plt.xlabel("Epochs")
        plt.ylabel("Değer")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close('all')
        return jsonify({"img": base64.b64encode(buf.getvalue()).decode()})

# ===================================================
# FULL TEST ANALİZİ
# ===================================================
@app.route("/full_test_analysis")
def full_test_analysis():
    global full_test_progress
    global full_test_results

    if model is None:
        print("Model yok.")
        return

    try:
        print("--- Analiz Başlatılıyor ---")
        test_csv = pd.read_csv("dataset/Test.csv")
        y_true, y_pred = [], []
        y_scores = []
        
        input_shape = model.input_shape[1:3]
        total = len(test_csv)
        full_test_progress = 0

        for idx, row in test_csv.iterrows():
            if idx % 20 == 0: time.sleep(0.005) 
            if idx % 2000 == 0: gc.collect()

            full_test_progress = int(((idx + 1) / total) * 95)
            
            try:
                img_path = os.path.join("dataset", row["Path"])
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, input_shape).astype("float32") / 255.0
                X = np.expand_dims(img, axis=0)
                
                preds = model.predict(X, verbose=0)[0]
                y_scores.append(preds)
                y_true.append(int(row["ClassId"]))
                y_pred.append(int(np.argmax(preds)))
            except: continue

        summary = []
        try:
            report = classification_report(y_true, y_pred, digits=4, output_dict=True)
            summary = [
                {"aciklama": "Doğruluk (%)", "deger": round(report["accuracy"] * 100, 2)},
                {"aciklama": "Ortalama Precision (%)", "deger": round(np.mean([v["precision"] for k,v in report.items() if k.isdigit()])*100,2)},
                {"aciklama": "Ortalama Recall (%)", "deger": round(np.mean([v["recall"] for k,v in report.items() if k.isdigit()])*100,2)},
                {"aciklama": "Ortalama F1-score (%)", "deger": round(np.mean([v["f1-score"] for k,v in report.items() if k.isdigit()])*100,2)},
            ]
        except Exception as e:
            print(f"İstatistik Hatası: {e}")

        cm_b64 = ""
        roc_b64 = ""
        
        try:
            with plot_lock:
                # CM
                plt.clf()
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=False, cmap="Blues")
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=90, bbox_inches='tight')
                plt.close('all')
                buf.seek(0)
                cm_b64 = base64.b64encode(buf.getvalue()).decode()

                # ROC
                n_classes = 43
                y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
                y_scores = np.array(y_scores)
                fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-average ROC (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Eğrisi (Tüm Test Seti)')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                buf_roc = io.BytesIO()
                plt.savefig(buf_roc, format="png", dpi=90, bbox_inches='tight')
                plt.close('all')
                buf_roc.seek(0)
                roc_b64 = base64.b64encode(buf_roc.getvalue()).decode()

        except Exception as e:
            print(f"Grafik Hatası: {e}")

        full_test_results = {"cm": cm_b64, "roc": roc_b64, "summary": summary}
        
        try:
            save_path = os.path.join("full_test_results.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(full_test_results, f, ensure_ascii=False)
            print(f"Sonuçlar dosyaya kaydedildi: {save_path}")
        except Exception as e:
            print(f"Dosya kaydetme hatası: {e}")

        print(f"Analiz Bitti.")
        full_test_progress = 100

    except Exception as e:
        print(f"GENEL HATA: {e}")
        full_test_progress = 100

@app.route("/full_test_analysis_start")
def start_full_test_analysis():
    threading.Thread(target=full_test_analysis).start()
    return jsonify({"status": "started"})

@app.route("/full_test_progress")
def full_test_progress_api():
    return jsonify({"progress": full_test_progress})

@app.route("/full_test_analysis_results")
def full_test_analysis_results():
    if not full_test_results:
        return jsonify({"cm": "", "roc": "", "summary": []})
    return jsonify(full_test_results)

@app.route("/load_saved_analysis")
def load_saved_analysis():
    try:
        save_path = os.path.join("full_test_results.json")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({}) 
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        return jsonify({})

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model yüklendi.")
    except: model = None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)