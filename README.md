# Chip-Fine-Tuning
模型微调需要基座模型配合推理，RAG受到知识匹配结果影响，虽然准确仍受限于基座模型token.在这个项目中提出一种全新的完全解耦的芯片植入式微调方法，将知识以芯片的形式插入模型推理过程中，芯片构建完全独立于基座，理论上可以为基座无限制的植入所需要的知识芯片。


目录结构：
knowledge_chip_system/
├── Dockerfile
├── requirements.txt
├── .gitlab-ci.yml          # CI/CD 自动构建、测试、发布
├── start.sh                # 生产启动脚本
├── config/                 # 配置中心
│   ├── __init__.py
│   ├── model_config.py
│   ├── chip_config.py
│   └── logging_config.py   # 日志配置
├── core/                   # 知识芯片核心
│   ├── __init__.py
│   ├── knowledge_chip.py
│   ├── router.py
│   ├── fusion.py
│   └── engine.py
├── models/                 # 大模型加载
│   ├── __init__.py
│   └── model_loader.py
├── training/               # 芯片训练（完全解耦）
│   ├── __init__.py
│   └── chip_trainer.py
├── service/                # API 服务
│   ├── __init__.py
│   └── server.py           # 集成日志 + 监控
├── utils/                  # 工具
│   ├── __init__.py
│   ├── logger.py           # 日志工具
│   └── version.py          # 版本管理
├── monitoring/             # 监控
│   ├── __init__.py
│   └── metrics.py          # Prometheus 指标
├── knowledge_chips/        # 芯片持久化目录（挂载）
└── logs/                   # 日志目录

