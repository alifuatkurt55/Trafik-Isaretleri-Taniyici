# 1. Python 3.10 imajını kullan
FROM python:3.10

# 2. Çalışma klasörünü ayarla
WORKDIR /code

# 3. Kütüphane listesini kopyala ve kur
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. GEREKLİ SİSTEM KÜTÜPHANELERİ + UNZIP ARACI
# Buraya 'unzip' komutunu ekledik
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 unzip

# 5. Kullanıcı ayarları
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 6. Kodları ve ZIP dosyasını kopyala
COPY --chown=user . $HOME/app

# 7. ZIP DOSYASINI AÇ (Otomatik Dataset Kurulumu) ⚡
# Eğer veri.zip varsa onu dataset klasörüne çıkarır
RUN if [ -f veri.zip ]; then unzip veri.zip -d dataset; fi

# 8. Uygulamayı başlat
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8"]
