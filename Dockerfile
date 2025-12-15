# 1. Python 3.9 imajını kullan
FROM python:3.9

# 2. Çalışma klasörünü ayarla
WORKDIR /code

# 3. Kütüphane listesini kopyala ve kur
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. OpenCV için gerekli sistem kütüphanelerini yükle (Çok Önemli!)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 5. Kullanıcı ayarları (Hugging Face güvenlik standardı)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 6. Kodları kopyala (Sahiplik ayarıyla birlikte)
COPY --chown=user . $HOME/app

# 7. Uygulamayı başlat (Lazy Loading yaptığımız için timeout sorun olmaz)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8"]
