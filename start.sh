#!/bin/bash

echo "--- AKILLI BAŞLATMA SENARYOSU (TRAIN/TEST AYRIMI) ---"

# 1. Klasörleri oluştur
mkdir -p dataset/Train
mkdir -p dataset/Test

# ----------------------------------------------------------------
# 2. ADIM: EĞİTİM (TRAIN) DOSYALARINI AÇMA
# ----------------------------------------------------------------
# İsmi 'train_' ile başlayan zip dosyaları aranıyor
count_train=$(ls train_*.zip 2>/dev/null | wc -l)

if [ "$count_train" != "0" ]; then
    echo "$count_train adet EĞİTİM (Train) paketi bulundu. dataset/Train içine açılıyor..."
    for f in train_*.zip; do
        echo "Açılıyor: $f"
        # -o: üzerine yaz, -q: sessiz, -d: hedef klasör
        unzip -o -q "$f" -d dataset/Train
    done
else
    echo "BİLGİ: Hiç eğitim (train_*.zip) paketi bulunamadı. Sadece test verisiyle devam edilecek."
fi

# ----------------------------------------------------------------
# 3. ADIM: TEST DOSYALARINI AÇMA
# ----------------------------------------------------------------
# İsmi 'train_' ile BAŞLAMAYAN zip dosyalarını bul (part1.zip, test.zip vb.)
# grep -v "train_" komutu "içinde train_ geçenleri hariç tut" demektir.
count_test=$(ls *.zip 2>/dev/null | grep -v "train_" | wc -l)

if [ "$count_test" != "0" ]; then
    echo "$count_test adet TEST paketi bulundu. dataset/Test içine açılıyor..."
    # Döngüde sadece train olmayanları seçiyoruz
    for f in *.zip; do
        if [[ "$f" != train_* ]]; then
            echo "Açılıyor: $f"
            unzip -o -q "$f" -d dataset/Test
        fi
    done
else
    echo "UYARI: Hiç test paketi bulunamadı!"
fi

# ----------------------------------------------------------------
# 4. ADIM: TEMİZLİK VE DÜZENLEME
# ----------------------------------------------------------------

# ZIP'lerden çıkan CSV'leri ve Meta klasörlerini ana klasöre taşı
echo "CSV ve Meta dosyaları düzenleniyor..."

# Train klasöründen çıkanlar
if ls dataset/Train/*.csv 1> /dev/null 2>&1; then mv dataset/Train/*.csv dataset/; fi
if [ -d "dataset/Train/Meta" ]; then mv dataset/Train/Meta dataset/; fi

# Test klasöründen çıkanlar
if ls dataset/Test/*.csv 1> /dev/null 2>&1; then mv dataset/Test/*.csv dataset/; fi
if [ -d "dataset/Test/Meta" ]; then mv dataset/Test/Meta dataset/; fi

# İç içe klasör oluştuysa düzelt (dataset/Test/Test veya dataset/Train/Train durumu)
if [ -d "dataset/Test/Test" ]; then
    echo "İç içe Test klasörü düzeltiliyor..."
    mv dataset/Test/Test/* dataset/Test/
    rmdir dataset/Test/Test
fi

if [ -d "dataset/Train/Train" ]; then
    echo "İç içe Train klasörü düzeltiliyor..."
    mv dataset/Train/Train/* dataset/Train/
    rmdir dataset/Train/Train
fi

# ----------------------------------------------------------------
# 5. ADIM: KONTROL VE BAŞLATMA
# ----------------------------------------------------------------
echo "--- dataset/Train İÇERİĞİ (İLK 5 DOSYA) ---"
ls dataset/Train | head -n 5
echo "--- dataset/Test İÇERİĞİ (İLK 5 DOSYA) ---"
ls dataset/Test | head -n 5
echo "--------------------------------------------"

echo "Uygulama başlatılıyor..."
# Timeout süresini 600 saniye (100 dakika) yaptık ki eğitim sırasında kapanmasın
exec gunicorn app:app --bind 0.0.0.0:7860 --workers 1 --threads 8 --timeout 6000
