#!/bin/bash

echo "--- BAŞLATMA SENARYOSU DEVREDE ---"

# 1. Klasör yapısını hazırla
mkdir -p dataset

# 2. ZIP dosyasını kontrol et ve aç
if [ -f "veri.zip" ]; then
    echo "veri.zip bulundu, açılıyor..."
    # -o: üzerine yaz, -q: sessiz ol, -d: hedef klasör
    unzip -o veri.zip -d dataset
    echo "ZIP açma işlemi tamamlandı."
else
    echo "UYARI: veri.zip bulunamadı!"
fi

# 3. Klasör yapısını kontrol et (Hata ayıklama için çok önemli)
echo "--- DATASET KLASÖR İÇERİĞİ (İLK 20 SATIR) ---"
ls -R dataset | head -n 20
echo "--------------------------------------------"

# 4. Eğer klasör yanlış çıktıysa (dataset/dataset/Test gibi) düzeltme yap
if [ -d "dataset/dataset/Test" ]; then
    echo "İç içe klasör tespit edildi, düzeltiliyor..."
    mv dataset/dataset/* dataset/
    rmdir dataset/dataset
fi

# 5. Uygulamayı başlat
echo "Uygulama başlatılıyor..."
exec gunicorn app:app --bind 0.0.0.0:7860 --workers 1 --threads 8 --timeout 120
