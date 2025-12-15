FROM python:3.10

WORKDIR /code

# Kütüphaneleri kur
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Sistem araçlarını kur (unzip ve opencv bağımlılıkları)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 unzip chmod

# Kullanıcı ayarları
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Dosyaları kopyala
COPY --chown=user . $HOME/app

# start.sh dosyasına çalıştırma izni ver (ÖNEMLİ)
RUN chmod +x $HOME/app/start.sh

# Başlatma komutunu start.sh olarak ayarla
CMD ["./start.sh"]
