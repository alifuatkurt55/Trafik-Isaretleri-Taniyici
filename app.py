# ============================================================
# KÜTÜPHANE İMPORTLARI (GEREKLİ MODÜLLERİN DAHİL EDİLMESİ)
# ============================================================
import json
import os  # İşletim sistemi işlemleri (dosya yolu bulma, klasör oluşturma vb.) için
import cv2  # OpenCV kütüphanesi: Görüntü işleme, okuma ve dönüştürme işlemleri için
import io  # Bellek içi (RAM) dosya işlemleri için (grafikleri diske kaydetmeden RAM'de tutmak için)
import base64  # Görüntü verilerini HTML img etiketinde göstermek üzere metin formatına çevirmek için
import time  # Zamanla ilgili işlemler (bekletme, süre ölçme vb.) için
import threading  # Çoklu iş parçacığı: Model eğitimini arka planda yapıp sunucuyu kilitlememek için
import joblib  # Eğitilen makine öğrenmesi modellerini (.pkl dosyası) kaydetmek ve yüklemek için
import numpy as np  # Sayısal hesaplamalar, matris işlemleri ve vektörler için
import pandas as pd  # Veri analizi ve CSV dosyalarını okumak/işlemek için
import matplotlib.pyplot as plt  # Grafik çizimi (plot) oluşturmak için
import seaborn as sns  # Daha gelişmiş ve estetik grafikler (örneğin Isı Haritası) çizmek için
from flask import Flask, render_template, jsonify, request  # Web sunucusu, HTML render ve API işlemleri için
from tqdm import tqdm  # Döngülerde ilerleme çubuğu göstermek için (Terminal tarafı için)
from skimage.feature import hog  # Görüntüden HOG (Histogram of Oriented Gradients) özelliklerini çıkarmak için
from sklearn.ensemble import RandomForestClassifier  # Sınıflandırma algoritması olarak Random Forest (Rastgele Orman) kullanımı
from sklearn.model_selection import train_test_split, learning_curve  # Veri bölme ve öğrenme eğrisi analizi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc  # Model başarısını ölçen metrikler
from sklearn.preprocessing import label_binarize  # ROC eğrisi çizimi için etiketleri ikili (binary) formata çevirme
import matplotlib  # Matplotlib'in genel ayarlarını değiştirmek için

# GUI (Pencere) hatası almamak için "Agg" backend'i kullanılır.
# Sunucularda ekran kartı/monitör olmadığı için grafiklerin arka planda çizilmesini sağlar.
matplotlib.use("Agg")

# Flask uygulamasını başlatıyoruz (__name__ mevcut dosyanın adını temsil eder)
app = Flask(__name__)

# ============================================================
# AYARLAR VE GLOBAL DEĞİŞKENLER
# ============================================================
# Eğitim sürecini web arayüzüne aktarmak için kullanılan durum sözlüğü
progress = {"percent": 0, "status": "", "running": False}

# Arayüzdeki "Loglar" kutusuna gönderilecek mesajları tutan liste
logs = []

# Eğitimi "Durdur" butonuna basıldığında kesmek için kullanılan kontrol değişkeni
stop_training = False

# Kodun çalıştığı dizini çalışma dizini olarak ayarla (Dosya yolu hatalarını önler)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Kullanıcının yükleyeceği resimlerin geçici olarak tutulacağı klasör yolu
UPLOAD_FOLDER = "dataset/Uploads"
# Eğer bu klasör yoksa oluştur (exist_ok=True: klasör varsa hata verme demektir)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Eğitilen modelin kaydedileceği dosya yolu
MODEL_PATH = "model/traffic_sign_rf.pkl" 

# ============================================================
# [YENİ] SINIF ETİKETLERİ SÖZLÜĞÜ (GTSRB Veri Seti)
# ============================================================
# Modelin tahmin ettiği 0-42 arası sayıların Türkçe karşılıkları
classes_dict = {
    0: 'Hız Limiti (20km/s)',
    1: 'Hız Limiti (30km/s)',
    2: 'Hız Limiti (50km/s)',
    3: 'Hız Limiti (60km/s)',
    4: 'Hız Limiti (70km/s)',
    5: 'Hız Limiti (80km/s)',
    6: 'Hız Limiti Sonu (80km/s)',
    7: 'Hız Limiti (100km/s)',
    8: 'Hız Limiti (120km/s)',
    9: 'Geçiş Yok',
    10: 'Kamyon Giremez',
    11: 'Öncelik Yolu',
    12: 'Anayol',
    13: 'Yol Ver',
    14: 'DUR',
    15: 'Taşıt Giremez',
    16: 'Kamyon Giremez (3.5 ton üstü)',
    17: 'Girişi Olmayan Yol',
    18: 'Genel Tehlike',
    19: 'Sola Keskin Viraj',
    20: 'Sağa Keskin Viraj',
    21: 'Çift Viraj',
    22: 'Engebeli Yol',
    23: 'Kaygan Yol',
    24: 'Sağdan Daralan Yol',
    25: 'Yol Çalışması',
    26: 'Trafik Işıkları',
    27: 'Yaya Geçidi',
    28: 'Okul Geçidi',
    29: 'Bisiklet Geçebilir',
    30: 'Buzlanma Tehlikesi',
    31: 'Vahşi Hayvan Çıkabilir',
    32: 'Hız Sınırı Sonu',
    33: 'Sağa Mecburi Yön',
    34: 'Sola Mecburi Yön',
    35: 'İleri Mecburi Yön',
    36: 'İleri veya Sağa Mecburi',
    37: 'İleri veya Sola Mecburi',
    38: 'Sağdan Gidiş',
    39: 'Soldan Gidiş',
    40: 'Ada Etrafında Dönüş',
    41: 'Geçiş Yasağı Sonu',
    42: 'Kamyon Geçiş Yasağı Sonu'
}

# ============================================================
# LOG FONKSİYONU
# ============================================================
def add_log(text):
    """
    Sisteme yeni bir log mesajı ekler ve bunu hem listeye hem de terminale yazar.
    Web arayüzü bu listeyi periyodik olarak okur.
    """
    global logs  # Global logs listesine erişim sağla
    logs.append(text)  # Listeye metni ekle
    print(text)  # Terminal ekranına da yaz (Debug için)

# ============================================================
# 1. ÖZELLİK ÇIKARMA (RENK + HOG)
# ============================================================
def extract_features(image):
    """
    Ham görüntüden modelin anlayabileceği sayısal verileri (vektörleri) çıkarır.
    Bu projede hem RENK (Color Histogram) hem de ŞEKİL (HOG) özellikleri birleştirilmiştir.
    """
    # Görüntüyü standart bir boyuta (32x32 piksel) getir. İşlem yükünü azaltır ve standartlaştırır.
    img_resized = cv2.resize(image, (32, 32))
    
    # --- Renk Özellikleri (HSV) ---
    # BGR formatından HSV (Hue, Saturation, Value) formatına geçiş yapıyoruz.
    # Trafik işaretlerinde renk (Kırmızı, Mavi) çok ayırt edicidir ve HSV ışıktan daha az etkilenir.
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Her kanal (H, S, V) için histogram (renk yoğunluk dağılımı) hesapla.
    # [8] parametresi "bin" sayısıdır; renk uzayını 8 parçaya bölerek özetler.
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180]) # Hue (Renk Özü)
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]) # Saturation (Doygunluk)
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]) # Value (Parlaklık)
    
    # Histogram değerlerini 0-1 arasına normalize et (Resim boyutu değişse de oranlar korunsun).
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    # Üç ayrı histogramı tek bir düz vektör haline getir (flatten).
    color_features = np.concatenate([hist_h, hist_s, hist_v]).flatten()

    # --- Şekil Özellikleri (HOG - Histogram of Oriented Gradients) ---
    # Şekil analizi için renk gerekmez, gri tonlamaya çevir.
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # HOG algoritması kenar yönlerini ve yoğunluklarını hesaplar.
    # orientations=8: 8 farklı yönde kenar ara.
    # pixels_per_cell=(4, 4): Görüntüyü 4x4'lük hücrelere böl.
    hog_features = hog(gray, orientations=8, pixels_per_cell=(4, 4),
                       cells_per_block=(2, 2), visualize=False)
    
    # Renk ve Şekil vektörlerini yan yana birleştir (hstack) ve tek bir "öznitelik vektörü" döndür.
    return np.hstack([color_features, hog_features])

# ============================================================
# 2. MODEL EĞİTİMİ FONKSİYONU
# ============================================================
def train_model():
    """
    Veri setini okur, özellikleri çıkarır, Random Forest modelini eğitir ve kaydeder.
    Bu fonksiyon uzun sürdüğü için Thread içinde çalıştırılır.
    """
    global progress, logs, stop_training, model
    stop_training = False  # Durdurma isteğini sıfırla
    logs = []  # Önceki logları temizle

    # Arayüze eğitimin başladığını bildir
    progress.update({"running": True, "percent": 0, "status": "Veri taranıyor..."})
    add_log("=== Model Eğitimi Başladı ===")

    data_dir = "dataset/Train"  # Eğitim verilerinin bulunduğu ana klasör
    image_paths, labels = [], []  # Resim yollarını ve etiketlerini tutacak boş listeler
    
    # Veri setindeki sınıf klasörlerini (0, 1, 2... 42) listele
    classes = os.listdir(data_dir)
    for label in classes:
        class_dir = os.path.join(data_dir, label)
        # Eğer okunan şey bir klasör değilse atla
        if not os.path.isdir(class_dir): continue
        
        # O sınıfın içindeki tüm resim dosyalarını gez
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name)) # Tam dosya yolu
            labels.append(int(label)) # Klasör adı etikettir (örn: '0')

    total_images = len(image_paths)
    add_log(f"Toplam {total_images} görüntü bulundu.")
    
    X, y = [], []  # X: Girdi verileri (Özellikler), y: Çıktı verileri (Etiketler)
    add_log("[1] Özellikler çıkarılıyor...")
    
    # Tüm resimleri tek tek işle
    for i, path in enumerate(image_paths):
        # Kullanıcı arayüzden "Durdur"a bastıysa döngüyü kır ve çık
        if stop_training: return

        img = cv2.imread(path) # Resmi diskten oku
        if img is None: continue # Resim bozuksa veya okunamadıysa atla

        # Özellik çıkarma fonksiyonunu çağır ve listeye ekle
        X.append(extract_features(img))
        y.append(labels[i])

        # Her 100 resimde bir ilerleme çubuğunu güncelle (Performans için her adımda yapmıyoruz)
        if i % 100 == 0:
            # İşlemin %0 ile %60 arası özellik çıkarma kısmıdır
            percent = int((i / total_images) * 60)
            progress.update({"percent": percent, "status": f"Özellik Çıkarımı: {i}/{total_images}"})

    # Listeleri Numpy dizisine çevir (Scikit-learn kütüphanesi numpy array ister)
    X = np.array(X)
    y = np.array(y)
    
    add_log("[2] Random Forest Eğitiliyor...")
    progress.update({"percent": 70, "status": "Eğitim başladı..."})

    # Veriyi %80 Eğitim (Train), %20 Doğrulama (Validation) olarak ayır.
    # random_state=42: Her seferinde aynı şekilde bölünmesi için sabit bir tohum değeri.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Sınıflandırıcısını tanımla
    # n_estimators=100: 100 adet karar ağacı kullanılacak.
    # n_jobs=-1: Bilgisayarın tüm işlemci çekirdeklerini kullan (Hızlandırır).
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    
    # Modeli eğitim verisiyle eğit (Fit işlemi)
    rf.fit(X_train, y_train)

    # Doğrulama verisi üzerinde başarı skorunu (Accuracy) hesapla
    acc = rf.score(X_val, y_val)
    add_log(f"Eğitim Tamamlandı. Doğruluk: %{acc*100:.2f}")

    add_log("[3] Kaydediliyor...")
    progress.update({"percent": 90, "status": "Kaydediliyor..."})
    
    os.makedirs("model", exist_ok=True) # Model klasörü yoksa oluştur
    joblib.dump(rf, MODEL_PATH) # Eğitilen modeli dosyaya kaydet
    
    # İLERİ SEVİYE ANALİZ İÇİN: ROC ve Learning Curve grafiklerinde kullanılmak üzere
    # doğrulama (validation) verisini de kaydediyoruz. (Eğitim verisi çok büyük olduğu için kaydetmiyoruz)
    joblib.dump((X_val, y_val), "model/val_data.pkl")

    # Bellekteki model değişkenini güncelle
    model = rf
    # Süreci tamamla
    progress.update({"percent": 100, "status": f"Bitti! Doğruluk: %{acc*100:.2f}", "running": False})

# ============================================================
# 3. YARDIMCI GÖRÜNTÜ İŞLEME FONKSİYONLARI
# ============================================================
def preprocess_smart(img):
    """
    Gelen resmi "sündürmeden" (aspect ratio bozmadan) kare formata getirir.
    Önce resmi küçültür, kalan boşlukları siyah ile doldurur (Padding).
    """
    h, w = img.shape[:2] # Resmin yükseklik ve genişliğini al
    target_size = 64 # Hedef kenar boyutu
    
    # En büyük kenara göre ölçekleme oranını bul
    scale = target_size / max(h, w)
    # Yeni boyutları hesapla
    new_w, new_h = int(w * scale), int(h * scale)
    # Resmi yeniden boyutlandır
    resized = cv2.resize(img, (new_w, new_h))
    
    # Hedef boyuta (64x64) ulaşmak için ne kadar boşluk kaldığını hesapla
    delta_w, delta_h = target_size - new_w, target_size - new_h
    # Boşluğu yukarı/aşağı ve sağ/sol olarak ikiye böl
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # OpenCV ile kenarlara siyah (value=[0,0,0]) çerçeve ekle
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Uygulama ilk açıldığında, eğer daha önce eğitilmiş model dosyası varsa onu yükle.
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None # Model yoksa None ata (kullanıcı önce eğitim yapmalı)

def predict_general(img_or_path, true_label=None):
    """
    Gelen bir resim verisi veya dosya yolu için tahmin işlemini yapar.
    DÜZELTME: Meta görseli artık gerçek etiketin değil, TAHMİN EDİLEN sınıfın görselidir.
    """
    if model is None: return {"error": "Model yok"} 
    
    # Görseli al
    if isinstance(img_or_path, str): img = cv2.imread(img_or_path)
    else: img = img_or_path 
    
    if img is None: return {"error": "Görsel yok"}

    # 1. Özellik Çıkarma ve Tahmin
    processed = preprocess_smart(img)
    feats = extract_features(processed)
    
    # Modelin Tahmini (Örn: 14)
    prediction_idx = int(model.predict([feats])[0])
    
    # Tahmin edilen sınıfın ismi (Örn: "DUR")
    class_name = classes_dict.get(prediction_idx, "Tanımsız İşaret")

    # 2. Test Görselini Base64 Yap (Web'de göstermek için)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".png", img_rgb)
    test_img_b64 = base64.b64encode(buffer).decode()

    # 3. META GÖRSELİ AYARLAMA (DÜZELTİLEN KISIM)
    # Varsayılan olarak meta resmi test resminin aynısı olsun (eğer meta bulunamazsa boş kalmasın)
    meta_b64 = test_img_b64 
    
    # Burada artık true_label'a DEĞİL, prediction_idx'e bakıyoruz.
    # Model ne tahmin ettiyse onun temiz grafiğini getiriyoruz.
    meta_path = os.path.join("dataset", "Meta", f"{prediction_idx}.png")
    
    if os.path.exists(meta_path):
        meta_img = cv2.imread(meta_path)
        # Meta resimler bazen RGBA (şeffaf) olabilir, onları da düzgün okuyalım
        if meta_img is not None:
            _, buf_meta = cv2.imencode(".png", cv2.cvtColor(meta_img, cv2.COLOR_BGR2RGB))
            meta_b64 = base64.b64encode(buf_meta).decode()

    # Sonuçları döndür
    return {
        "prediction": prediction_idx, 
        "prediction_text": class_name,
        "true_label": true_label if true_label is not None else "Bilinmiyor", 
        "test_img": test_img_b64, 
        "meta_img": meta_b64 # Artık tahmin edilen sınıfın resmi
    }

# ============================================================
# WEB ROUTES (URL YÖNLENDİRMELERİ)
# ============================================================
@app.route("/")
def index():
    # Ana sayfa istendiğinde 'index.html' dosyasını kullanıcıya gönder
    return render_template("index.html")

@app.route("/start_training")
def start_training():
    """Eğitimi başlatan API."""
    global stop_training
    # Eğer eğitim zaten çalışıyorsa tekrar başlatma
    if progress["running"]: return "Çalışıyor"
    stop_training = False
    # Eğitimi ayrı bir Thread (iş parçacığı) olarak başlat ki arayüz donmasın
    threading.Thread(target=train_model).start()
    return "Başlatıldı"

@app.route("/stop_training")
def stop_training_route():
    """Eğitimi durduran API."""
    global stop_training
    stop_training = True # Döngüleri kıracak bayrağı aktif et
    return "Durduruldu"

@app.route("/progress")
def get_progress():
    # Frontend (JavaScript) bu adresi sürekli sorgular, ilerleme durumunu JSON olarak alır
    return jsonify(progress)

@app.route("/logs")
def get_logs():
    # Log mesajlarını JSON olarak frontend'e gönderir
    return jsonify({"logs": logs})

@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    """Kullanıcının "Sürükle Bırak" ile yüklediği resmi tahmin eder."""
    # İstekte dosya var mı kontrol et
    if 'image' not in request.files: return jsonify({"error": "Dosya yok"})
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "Seçilmedi"})
    
    # Dosyayı diske kaydetmeden direkt RAM'den oku
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Genel tahmin fonksiyonunu kullan (Artık isim de döndürüyor)
    result = predict_general(img, true_label=None)
    
    # Ekranda "Bilinmiyor" yerine "Kullanıcı Yüklemesi" yazsın
    if result.get("true_label") == "Bilinmiyor": result["true_label"] = "Kullanıcı Yüklemesi"
    
    # JSON olarak döndür: {prediction: 14, prediction_text: "DUR", ...}
    return jsonify(result)

@app.route("/predict_random_test")
def predict_random_test():
    """Test veri setinden rastgele bir resim seçer ve tahmin eder."""
    # Test.csv dosyasını oku ve rastgele 1 satır seç
    row = pd.read_csv("dataset/Test.csv").sample(1).iloc[0]
    # Path (yol) ve ClassId (gerçek sınıf) bilgilerini alarak tahmin yap
    return jsonify(predict_general(os.path.join("dataset", row["Path"]), int(row["ClassId"])))

# ============================================================
# TOPLU TEST FONKSİYONU
# ============================================================
@app.route("/predict_batch_test/<int:num_samples>")
def predict_batch_test(num_samples):
    """
    Belirtilen sayıda (örn: 50) rastgele resmi Test veri setinden çeker ve
    hepsini tek tek tahmin ederek sonuçları bir liste halinde döner.
    """
    test_csv = pd.read_csv(os.path.join("dataset", "Test.csv"))
    samples = test_csv.sample(num_samples) # Rastgele <num_samples> kadar satır seç

    results = []
    for _, row in samples.iterrows():
        img_path = os.path.join("dataset", row["Path"])
        true_label = int(row["ClassId"])
        # Tahmin yap
        result = predict_general(img_path, true_label)
        results.append(result)

    return jsonify(results)

# ============================================================
# GRAFİK ANALİZ FONKSİYONLARI (CONFUSION MATRIX, ROC, LEARNING CURVE)
# ============================================================
@app.route("/confusion_samples/<int:num_samples>")
def confusion_samples(num_samples=30):
    """
    Rastgele seçilen 30-50 örnek üzerinde mini bir Confusion Matrix (Karmaşıklık Matrisi) çizer.
    Modelin hangi sınıfları karıştırdığını hızlıca görmek için kullanılır.
    """
    dataset_dir = "dataset"
    test_csv_path = os.path.join(dataset_dir, "Test.csv")
    model_path = os.path.join("model", "traffic_sign_rf.pkl") # Random Forest model yolu

    test_data = pd.read_csv(test_csv_path)
    
    # Modeli yerel olarak yükle (Global model yerine anlık dosya kontrolü)
    if not os.path.exists(model_path): return jsonify({"error": "Model bulunamadı"})
    local_model = joblib.load(model_path)

    # Rastgele örnek seç
    sample_data = test_data.sample(n=num_samples)
    true_labels, pred_labels = [], []

    # Seçilen örnekleri tahmin et
    for row in sample_data.itertuples():
        img_path = os.path.join(dataset_dir, row.Path)
        img = cv2.imread(img_path)
        if img is None: continue

        # Hızlı işlem için burada manuel özellik çıkarma yapılmış (Ana fonksiyonu kullanmak daha iyi olurdu)
        # Not: Tutarlılık için preprocess_smart ve extract_features kullanılmalı, ancak kod orijinaline sadık kalındı.
        processed = preprocess_smart(img)
        features = extract_features(processed)
        pred = local_model.predict([features])[0]

        true_labels.append(int(row.ClassId))
        pred_labels.append(pred)

    # --------------------------------------------------------------------------
    # [GÜNCELLEME] İsimleri göstermek için etiket hazırlığı
    # --------------------------------------------------------------------------
    # 1. Sadece bu örneklemde geçen benzersiz sınıfları bul ve sırala
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    
    # 2. Bu numaralara (0, 14, 25...) karşılık gelen isimleri sözlükten çek
    tick_labels = [classes_dict.get(lbl, str(lbl)) for lbl in unique_labels]

    # Confusion Matrix Hesapla (Gerçek vs Tahmin Edilen)
    # labels=unique_labels diyerek matrisin satır/sütun sırasını sabitliyoruz
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    num_classes = cm.shape[0]

    # Grafik boyutunu sınıf sayısına göre dinamik ayarla (İsimler sığsın diye biraz genişlettik)
    fig_size = min(max(10, num_classes // 1.5), 40)
    plt.figure(figsize=(fig_size, fig_size))
    
    # Isı haritası (Heatmap) çiz
    # xticklabels ve yticklabels ile eksenlere isimleri atıyoruz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
        xticklabels=tick_labels, yticklabels=tick_labels,
        annot_kws={"fontsize": max(4, 12 - num_classes//20)})
        
    plt.xlabel("Predicted (Tahmin Edilen)")
    plt.ylabel("True (Gerçek)")
    plt.title(f"Confusion Matrix ({num_samples} Örnek)")

    # İsimler uzun olduğu için okunaklı olsun diye 45 derece eğik yazdırıyoruz
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Grafiği belleğe (RAM) kaydet
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    # Grafiği Base64 formatına çevirip frontend'e gönder
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    
    return jsonify({"img": img_base64})

@app.route("/roc_plot")
def roc_plot():
    """
    Çok Sınıflı ROC Eğrisi (Micro-Average) çizer.
    Bu grafik, modelin "Doğru Pozitif" yakalama oranı ile "Yanlış Pozitif" oranı arasındaki dengeyi gösterir.
    Eğri sol üste ne kadar yakınsa model o kadar iyidir.
    """
    # Eğitim sırasında kaydedilen doğrulama verisi var mı kontrol et
    if not os.path.exists("model/val_data.pkl") or model is None:
        return jsonify({"error": "Analiz için eğitim verisi bulunamadı. Lütfen modeli tekrar eğitin."})

    # Veriyi yükle
    X_val, y_val = joblib.load("model/val_data.pkl")
    
    # Etiketleri binarize et (One-vs-Rest mantığı: Her sınıf için "Bu sınıf mı? Evet/Hayır" formatı)
    classes = model.classes_
    y_val_bin = label_binarize(y_val, classes=classes)

    # Modelin her sınıf için verdiği olasılık değerlerini al
    y_score = model.predict_proba(X_val)

    # ROC eğrisi verilerini hesapla (Micro-average yöntemi ile genel başarı)
    fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr) # Eğri altında kalan alan (Area Under Curve)

    # Grafiği çiz
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Micro-average ROC (AUC = {roc_auc:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Rastgele tahmin çizgisi (Referans)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Pozitif)')
    plt.ylabel('True Positive Rate (Doğru Pozitif)')
    plt.title('ROC Eğrisi (Micro-Average)')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Görseli Base64 yap ve gönder
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return jsonify({"img": base64.b64encode(buf.getvalue()).decode()})

@app.route("/learning_curve_plot")
def learning_curve_plot():
    """
    Öğrenme Eğrisi (Learning Curve):
    Modelin veri miktarı arttıkça başarısının nasıl değiştiğini gösterir.
    Overfitting (Ezberleme) veya Underfitting (Öğrenememe) durumlarını tespit etmek için kritiktir.
    """
    if not os.path.exists("model/val_data.pkl") or model is None:
        return jsonify({"error": "Veri yok. Lütfen eğitim yapın."})
    
    X_val, y_val = joblib.load("model/val_data.pkl")
    
    # learning_curve fonksiyonu veriyi kademeli olarak artırarak (örn: %10, %50, %100) eğitim yapar.
    # cv=3: 3 katlı çapraz doğrulama kullanır.
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_val, y_val, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), # 5 farklı boyutta test et
        scoring="accuracy"
    )

    # Skorların ortalamasını al
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Grafiği çiz
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score (Eğitim Başarısı)")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score (Test Başarısı)")
    
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

# ============================================================
# TAM TEST ANALİZİ VE KAYIT YÖNETİMİ
# ============================================================

# Analiz sonuçlarının kaydedileceği dosya yolu
ANALYSIS_FILE = "model/last_full_analysis.json"

@app.route("/full_test_analysis")
def full_test_analysis():
    """
    Test.csv üzerindeki TÜM verileri test eder, sonucu JSON dosyasına KAYDEDER ve döner.
    """
    if model is None: return jsonify({"error": "Model yok"})
    
    test_csv = pd.read_csv("dataset/Test.csv")
    y_true, y_pred = [], []
    read_ok = 0
    
    # Tüm test verisini döngüye al
    for _, row in test_csv.iterrows():
        img_path = os.path.join("dataset", row["Path"])
        img = cv2.imread(img_path)
        if img is None: continue
        read_ok += 1
        
        # Ön işleme ve özellik çıkarma
        processed = preprocess_smart(img)
        feats = extract_features(processed)
        
        # Gerçek ve Tahmin edilen değerleri listelere ekle
        y_true.append(int(row["ClassId"]))
        y_pred.append(int(model.predict([feats])[0]))
        
    # Doğruluk oranını hesapla
    acc = accuracy_score(y_true, y_pred)
    
    # --- Rapor Oluşturma ---
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    report_list = []
    for key, value in report_dict.items():
        if key.isdigit():
            class_id = int(key)
            class_name = classes_dict.get(class_id, "Tanımsız")
            report_list.append({
                "id": class_id,
                "name": class_name,
                "precision": f"{value['precision']:.2f}",
                "recall": f"{value['recall']:.2f}",
                "f1": f"{value['f1-score']:.2f}",
                "support": value['support']
            })

    # --- Grafik Oluşturma ---
    # Not: Full analizde tüm sınıf isimlerini yazmak grafiği çok sıkıştırabilir,
    # bu yüzden burada sadece genel yoğunluk haritası (heatmap) bırakıldı.
    # İsterseniz önceki kodumuzdaki gibi xticklabels ekleyebilirsiniz.
    # --- Grafik Oluşturma (Confusion Samples Stiline Uyarlı) ---
    
    # 1. Tüm sınıfları (0-42) sıralı olarak alıyoruz (Tam analiz olduğu için hepsi görünmeli)
    all_classes = sorted(classes_dict.keys())
    
    # 2. Bu numaralara karşılık gelen isimleri sözlükten çek
    tick_labels = [classes_dict.get(i, str(i)) for i in all_classes]

    # 3. Matrisi hesapla (labels parametresi ile sırayı sabitliyoruz)
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    
    num_classes = cm.shape[0] # Genelde 43 olacaktır

    # Grafik boyutunu dinamik ayarla (43 sınıf olduğu için büyük olmalı, örn: 24x24)
    # confusion_samples'taki mantığın aynısı, sadece minimum değeri büyüttük.
    fig_size = max(20, num_classes // 1.5) 
    plt.figure(figsize=(fig_size, fig_size))
    
    # Isı haritası (Heatmap) çiz
    # xticklabels ve yticklabels ile eksenlere İSİMLERİ atıyoruz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
        xticklabels=tick_labels, yticklabels=tick_labels,
        annot_kws={"fontsize": 9}) # Yazı boyutu (Tüm tabloya sığması için küçülttük)
        
    plt.xlabel("Predicted (Tahmin Edilen)", fontsize=14)
    plt.ylabel("True (Gerçek)", fontsize=14)
    plt.title(f"Confusion Matrix (Acc: %{acc*100:.2f})", fontsize=16)

    # İsimler uzun olduğu için okunaklı olsun diye 45/90 derece eğik yazdırıyoruz
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, fontsize=10)

    # Grafiği belleğe (RAM) kaydet
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    cm_b64 = base64.b64encode(buf.getvalue()).decode()
    
    # --- SONUCU HAZIRLA VE KAYDET ---
    result_data = {
        "cm": cm_b64,
        "summary": [
            {"aciklama": "Test Sayısı", "deger": read_ok}, 
            {"aciklama": "Doğruluk", "deger": f"%{acc*100:.2f}"}
        ],
        "report": report_list
    }
    
    # Sonuçları JSON dosyasına yaz
    try:
        with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False)
    except Exception as e:
        print(f"Kayıt hatası: {e}")

    return jsonify(result_data)

@app.route("/load_last_analysis")
def load_last_analysis():
    """
    Daha önce yapılmış ve kaydedilmiş analizi diskten okur.
    Yeniden hesaplama yapmaz, çok hızlıdır.
    """
    if not os.path.exists(ANALYSIS_FILE):
        return jsonify({"error": "Kayıtlı analiz bulunamadı. Önce 'Tam Analizi Başlat' butonuna basın."})
    
    try:
        with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Dosya okunamadı: {e}"})

# ============================================================
# PROGRAM BAŞLANGIÇ NOKTASI
# ============================================================
if __name__ == "__main__":
    # Ortam değişkenlerinden (Environment Variables) PORT bilgisini al, yoksa 5000'i kullan
    port = int(os.environ.get("PORT", 5000))
    # Flask uygulamasını başlat
    # host='0.0.0.0': Tüm ağ arayüzlerinden erişime izin ver (Localhost dışından da erişilebilir)
    app.run(host='0.0.0.0', port=port)
