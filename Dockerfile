FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY requirements.txt .
RUN apt update && apt install -y python3-pip && pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python3", "-m", "service.server"]