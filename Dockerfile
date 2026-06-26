# Headless-Training für den Mario-RL-Agenten.
# Basis-Image bringt PyTorch + CUDA passend zur lokalen Version (torch 2.6 / cu124) mit.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# System-Abhängigkeiten für OpenCV (libGL etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Zuerst nur die Requirements kopieren -> besseres Layer-Caching beim Neubauen.
# torch/torchvision sind im Basis-Image bereits vorhanden und werden nicht neu installiert.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Standard: Training ohne Live-Fenster (im Container gibt es kein Display).
# Episodenzahl o. Ä. lassen sich beim Start überschreiben:
#   docker run --gpus all mario-rl python train.py --no-render --episodes 2000
CMD ["python", "train.py", "--no-render"]
