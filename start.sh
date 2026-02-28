# 构建镜像
docker build -t kmodel-api:v1 .

# 运行（GPU）
docker run -d \
  --gpus all \
  -p 7860:7860 \
  -v $(pwd)/knowledge_chips:/app/knowledge_chips \
  --restart always \
  --name kmodel-service \
  kmodel-api:v1