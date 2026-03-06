# 直接使用 Ubuntu 24.04 官方基础镜像
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装 OpenGL、X11 库以及 UE5 运行必需的系统组件
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    # 核心图形库 (24.04 标准)
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglapi-mesa \
    # UE5 运行必需依赖
    libpulse0 \
    libasound2t64 \
    libnss3 \
    libxcomposite1 \
    libxcursor1 \
    libxi6 \
    libxtst6 \
    libxrandr2 \
    libxdamage1 \
    libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*
# 设置环境变量，告知系统使用 NVIDIA 作为 OpenGL 后端
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

WORKDIR /simworld
