#!/bin/bash

echo "--- DÜZELTME SENARYOSU DEVREDE ---"

# 1. Klasör yapısını oluştur (Test klasörünü biz elle açıyoruz)
mkdir -p dataset/Test

# 2. ZIP dosyalarını dataset/Test içine aç
count=$(ls *.zip 2>/dev/null | wc -l)

if [ "$count" != "0" ]; then
    echo "$count adet ZIP dosyası bulundu, dataset/Test içine açılıyor..."
    for f in *.zip; do
        # -q: sessiz mod (logları şişirmesin diye)
        # -d dataset/Test: Dosyaları Test klasörünün içine zorla
        unzip -o -q "$f" -d dataset/Test
    done
    echo "ZIP açma işlemi tamamlandı."
else
    echo "UYARI: Hiç ZIP dosyası bulunamadı!"
fi

# 3. CSV ve Meta Dosyalarını Düzeltme (ÖNEMLİ!)
# Eğer ZIP içinde Test.csv de varsa, o da dataset/Test içine girmiştir.
# Ama kod muhtemelen onu 'dataset/Test.csv' olarak arıyordur.
# O yüzden .csv dosyalarını ve Meta klasörünü bir üst kata taşıyoruz.

if ls dataset/Test/*.csv 1> /dev/null 2>&1; then
    echo "CSV dosyaları ana dataset klasörüne taşınıyor..."
    mv dataset/Test/*.csv dataset/
fi

if [ -d "dataset/Test/Meta" ]; then
    echo "Meta klasörü ana dataset klasörüne taşınıyor..."
    mv dataset/Test/Meta dataset/
fi

# 4. Klasör Kontrolü (Loglarda görelim)
echo "--- dataset/Test İÇERİĞİ (İLK 10 DOSYA) ---"
ls dataset/Test | head -n 10
echo "--- dataset ANA KLASÖR İÇERİĞİ ---"
ls dataset
echo "--------------------------------------------"

# 5. İç içe klasör oluştuysa düzelt (dataset/Test/Test durumu)
if [ -d "dataset/Test/Test" ]; then
    echo "İç içe Test klasörü tespit edildi, düzeltiliyor..."
    mv dataset/Test/Test/* dataset/Test/
    rmdir dataset/Test/Test
fi

# 6. Uygulamayı başlat
echo "Uygulama başlatılıyor..."
exec gunicorn app:app --bind 0.0.0.0:7860 --workers 1 --threads 8 --timeout 120
