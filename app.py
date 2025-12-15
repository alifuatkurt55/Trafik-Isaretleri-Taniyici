import os
import io
import base64
import time
import threading
from flask import Flask, render_template, jsonify, request

# ============================================================
# 1. OPTİMİZASYON AYARLARI (LAZY LOADING İÇİN)
# ============================================================
# Matplotlib backend ayarını en başta yapıyoruz ama pyplot'u import etmiyoruz.
import matplotlib
matplotlib.use("Agg")

app = Flask(__name__)

# ============================================================
# AYARLAR VE GLOBAL DEĞİŞKENLER
# ============================================================
progress = {"percent": 0, "status": "", "running": False}
logs = []
stop_training = False

# Çalışma dizini ayarı
os.chdir(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER = "dataset/Uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = "model/traffic_sign_rf.pkl" 

# Modeli başlangıçta None yapıyoruz (RAM tasarrufu)
model = None

# ============================================================
# LOG FONKSİYONU
# ============================================================
def add_log(text):
    global logs
    logs.append(text)
    print(text)

# ============================================================
# YARDIMCI: MODEL YÜKLEME (Sadece ihtiyaç anında çalışır)
# ============================================================
def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            import joblib  # Lazy import
            try:
                model = joblib.load(MODEL_PATH)
                print("Model RAM'e yüklendi.")
            except Exception as e:
                print(f"Model yükleme hatası: {e}")
                return None
    return model

# ============================================================
# 2. ÖZELLİK ÇIKARMA (RENK + HOG)
# ============================================================
def extract_features(image):
    # İthalatları fonksiyon içine aldık (Lazy Import)
    import cv2
    import numpy as np
    from skimage.feature import hog

    img_resized = cv2.resize(image, (32, 32))
    
    # --- Renk (HSV) ---
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    color_features = np.concatenate([hist_h, hist_s, hist_v]).flatten()

    # --- Şekil (HOG) ---
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2), visualize=False)
    
    return np.hstack([color_features, hog_features])

# ============================================================
# 3. MODEL EĞİTİMİ
# ============================================================
def train_model():
    global progress, logs, stop_training, model
    
    # Ağır kütüphaneler sadece eğitim başladığında yüklenir
    import cv2
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    stop_training = False
    logs = []

    progress.update({"running": True, "percent": 0, "status": "Veri taranıyor..."})
    add_log("=== Model Eğitimi Başladı ===")

    data_dir = "dataset/Train"
    image_paths, labels = [], []
    
    if not os.path.exists(data_dir):
        add_log("Hata: 'dataset/Train' klasörü bulunamadı!")
        progress.update({"running": False})
        return

    classes = os.listdir(data_dir)
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir): continue
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(int(label))

    total_images = len(image_paths)
    add_log(f"Toplam {total_images} görüntü bulundu.")
    
    X, y = [], []
    add_log("[1] Özellikler çıkarılıyor...")
    
    for i, path in enumerate(image_paths):
        if stop_training: return

        img = cv2.imread(path)
        if img is None: continue

        X.append(extract_features(img))
        y.append(labels[i])

        if i % 100 == 0:
            percent = int((i / total_images) * 60)
            progress.update({"percent": percent, "status": f"Özellik Çıkarımı: {i}/{total_images}"})

    X = np.array(X)
    y = np.array(y)
    
    add_log("[2] Random Forest Eğitiliyor...")
    progress.update({"percent": 70, "status": "Eğitim başladı..."})

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    acc = rf.score(X_val, y_val)
    add_log(f"Eğitim Tamamlandı. Doğruluk: %{acc*100:.2f}")

    add_log("[3] Kaydediliyor...")
    progress.update({"percent": 90, "status": "Kaydediliyor..."})
    
    os.makedirs("model", exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    
    # Eğitim verilerini analiz için saklayalım
    joblib.dump((X_val, y_val), "model/val_data.pkl")

    model = rf
    progress.update({"percent": 100, "status": f"Bitti! Doğruluk: %{acc*100:.2f}", "running": False})

# ============================================================
# 4. GÖRÜNTÜ İŞLEME YARDIMCISI
# ============================================================
def preprocess_smart(img):
    import cv2 # Lazy Import
    h, w = img.shape[:2]
    target_size = 64
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    delta_w, delta_h = target_size - new_w, target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def predict_general(img_or_path, true_label=None):
    import cv2 # Lazy Import
    
    # Modeli al (Yüklü değilse şimdi yükler)
    current_model = get_model()
    if current_model is None: return {"error": "Model bulunamadı veya yüklenemedi."}

    if isinstance(img_or_path, str): img = cv2.imread(img_or_path)
    else: img = img_or_path
    if img is None: return {"error": "Görsel okunamadı"}

    processed = preprocess_smart(img)
    feats = extract_features(processed)
    prediction = int(current_model.predict([feats])[0])

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".png", img_rgb)
    test_img_b64 = base64.b64encode(buffer).decode()

    meta_b64 = test_img_b64
    if true_label is not None and isinstance(true_label, int):
        meta_path = os.path.join("dataset", "Meta", f"{true_label}.png")
        if os.path.exists(meta_path):
            meta_img = cv2.imread(meta_path)
            _, buf_meta = cv2.imencode(".png", cv2.cvtColor(meta_img, cv2.COLOR_BGR2RGB))
            meta_b64 = base64.b64encode(buf_meta).decode()

    return {"prediction": prediction, "true_label": true_label if true_label is not None else "Bilinmiyor", "test_img": test_img_b64, "meta_img": meta_b64}

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index(): return render_template("index.html")

@app.route("/start_training")
def start_training():
    global stop_training
    if progress["running"]: return "Çalışıyor"
    stop_training = False
    threading.Thread(target=train_model).start()
    return "Başlatıldı"

@app.route("/stop_training")
def stop_training_route():
    global stop_training
    stop_training = True
    return "Durduruldu"

@app.route("/progress")
def get_progress(): return jsonify(progress)

@app.route("/logs")
def get_logs(): return jsonify({"logs": logs})

@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    import cv2
    import numpy as np

    if 'image' not in request.files: return jsonify({"error": "Dosya yok"})
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "Seçilmedi"})
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    result = predict_general(img, true_label=None)
    if result.get("true_label") == "Bilinmiyor": result["true_label"] = "Kullanıcı Yüklemesi"
    return jsonify(result)

@app.route("/predict_random_test")
def predict_random_test():
    import pandas as pd # Lazy Import
    try:
        row = pd.read_csv("dataset/Test.csv").sample(1).iloc[0]
        return jsonify(predict_general(os.path.join("dataset", row["Path"]), int(row["ClassId"])))
    except Exception as e:
        return jsonify({"error": f"Test verisi okunamadı: {str(e)}"})

@app.route("/predict_batch_test/<int:num_samples>")
def predict_batch_test(num_samples):
    import pandas as pd # Lazy Import
    test_csv = pd.read_csv(os.path.join("dataset", "Test.csv"))
    samples = test_csv.sample(num_samples)

    results = []
    for _, row in samples.iterrows():
        img_path = os.path.join("dataset", row["Path"])
        true_label = int(row["ClassId"])
        result = predict_general(img_path, true_label)
        results.append(result)

    return jsonify(results)

@app.route("/confusion_samples/<int:num_samples>")
def confusion_samples(num_samples=30):
    # Bu fonksiyon için gerekli importlar
    import cv2
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from skimage.feature import hog
    from sklearn.metrics import confusion_matrix
    
    dataset_dir = "dataset"
    test_csv_path = os.path.join(dataset_dir, "Test.csv")
    
    # Model kontrolü
    current_model = get_model()
    if current_model is None: return jsonify({"error": "Model yok"})

    test_data = pd.read_csv(test_csv_path)

    # Rastgele benzersiz örnekler
    sample_data = test_data.sample(n=num_samples)

    true_labels, pred_labels = [], []

    for row in sample_data.itertuples():
        img_path = os.path.join(dataset_dir, row.Path)
        img = cv2.imread(img_path)
        if img is None: continue

        # extract_features fonksiyonunu kullanmak daha güvenli
        # ama orijinal koddaki gibi manuel de yapılabilir, burada extract_features kullanıyorum:
        feats = extract_features(preprocess_smart(img))
        pred = current_model.predict([feats])[0]

        true_labels.append(int(row.ClassId))
        pred_labels.append(pred)

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    num_classes = cm.shape[0]

    # Figure boyutunu dinamik ayarlama
    fig_size = min(max(8, num_classes // 2), 40)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
        annot_kws={"fontsize": max(4, 12 - num_classes//20)})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({num_samples} Örnek)")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    
    return jsonify({"img": img_base64})

@app.route("/roc_plot")
def roc_plot():
    # Lazy Imports
    import matplotlib.pyplot as plt
    import joblib
    import numpy as np
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    """Çok Sınıflı ROC Eğrisi (Macro-Average)"""
    current_model = get_model()
    
    if not os.path.exists("model/val_data.pkl") or current_model is None:
        return jsonify({"error": "Analiz için eğitim verisi bulunamadı. Lütfen modeli tekrar eğitin."})

    # Kaydedilmiş doğrulama verisini yükle
    X_val, y_val = joblib.load("model/val_data.pkl")
    
    # Sınıfları binarize et (One-vs-Rest mantığı için)
    classes = current_model.classes_
    y_val_bin = label_binarize(y_val, classes=classes)
    
    # Olasılıkları al
    try:
        y_score = current_model.predict_proba(X_val)
    except:
        return jsonify({"error": "Bu model olasılık tahmini (predict_proba) desteklemiyor."})

    # ROC Eğrilerini hesapla
    fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Micro-average ROC (AUC = {roc_auc:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Pozitif)')
    plt.ylabel('True Positive Rate (Doğru Pozitif)')
    plt.title('ROC Eğrisi (Micro-Average)')
    plt.legend(loc="lower right")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return jsonify({"img": base64.b64encode(buf.getvalue()).decode()})

@app.route("/learning_curve_plot")
def learning_curve_plot():
    import matplotlib.pyplot as plt
    import numpy as np
    import joblib
    from sklearn.model_selection import learning_curve

    current_model = get_model()
    if not os.path.exists("model/val_data.pkl") or current_model is None:
        return jsonify({"error": "Veri yok. Lütfen eğitim yapın."})
    
    X_val, y_val = joblib.load("model/val_data.pkl")
    
    train_sizes, train_scores, test_scores = learning_curve(
        current_model, X_val, y_val, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy"
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.title("Öğrenme Eğrisi (Learning Curve)")
    plt.xlabel("Eğitim Örnek Sayısı")
    plt.ylabel("Doğruluk (Accuracy)")
    plt.legend(loc="best")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return jsonify({"img": base64.b64encode(buf.getvalue()).decode()})

@app.route("/full_test_analysis")
def full_test_analysis():
    import cv2
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    current_model = get_model()
    if current_model is None: return jsonify({"error": "Model yok"})
    
    test_csv = pd.read_csv("dataset/Test.csv")
    y_true, y_pred = [], []
    read_ok = 0
    
    for _, row in test_csv.iterrows():
        img = cv2.imread(os.path.join("dataset", row["Path"]))
        if img is None: continue
        read_ok += 1
        processed = preprocess_smart(img)
        feats = extract_features(processed)
        y_true.append(int(row["ClassId"]))
        y_pred.append(int(current_model.predict([feats])[0]))
        
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (Acc: %{acc*100:.2f})")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    
    return jsonify({
        "cm": base64.b64encode(buf.getvalue()).decode(),
        "summary": [{"aciklama": "Test Sayısı", "deger": read_ok}, {"aciklama": "Doğruluk", "deger": f"%{acc*100:.2f}"}]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
