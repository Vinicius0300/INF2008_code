# Use Python 3.10.4 como base
FROM python:3.10.4-slim

# Instala dependências do sistema necessárias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho com o nome do projeto
WORKDIR /vfss-data-split

# Copia o arquivo requirements.txt
COPY requirements.txt .

# Instala as dependências com timeout maior e índice extra para PyTorch CPU
RUN pip install --no-cache-dir --timeout 1000 --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copia todo o conteúdo do repositório
COPY . .

# Define o diretório de trabalho
WORKDIR /vfss-data-split

# Define variável de ambiente
ENV PYTHONUNBUFFERED=1

# Define o entrypoint e comando padrão
ENTRYPOINT ["python", "aplicacao-unet.py"]
CMD []