#!/bin/bash

echo "--- BAŞLATMA SENARYOSU (MULTI-ZIP) DEVREDE ---"

# 1. Klasör yapısını hazırla
mkdir -p dataset

# 2. Tüm ZIP dosyalarını bul ve sırayla aç
# part1.zip, part2.zip vs. hepsini tek tek açar
count=$(ls *.zip 2>/dev/null | wc -l)

if [ "$count" != "0" ]; then
    echo "$count adet ZIP dosyası bulundu, açılıyor..."
    for f in *.zip; do
        echo "Açılıyor: $f"
        unzip -o "$f" -d dataset
    done
    echo "Tüm ZIP açma işlemleri tamamlandı."
else
    echo "UYARI: Hiç ZIP dosyası bulunamadı!"
fi

# 3. Klasör yapısını kontrol et
echo "--- DATASET KLASÖR İÇERİĞİ (İLK 20 SATIR) ---"
ls -R dataset | head -n 20
echo "--------------------------------------------"

# 4. İç içe klasör düzeltmesi (Eğer yanlışlıkla dataset/dataset olursa)
if [ -d "dataset/dataset/Test" ]; then
    echo "İç içe klasör tespit edildi, düzeltiliyor..."
    mv dataset/dataset/* dataset/
    rmdir dataset/dataset
fi

# 5. Uygulamayı başlat
echo "Uygulama başlatılıyor..."
exec gunicorn app:app --bind 0.0.0.0:7860 --workers 1 --threads 8 --timeout 120
