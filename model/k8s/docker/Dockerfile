FROM rayproject/ray:2.46.0-py310-aarch64

USER root

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libxi6 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxinerama1 \
    libgtk-3-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ldconfig

ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV OPENCV_IO_ENABLE_JASPER=0
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Agg

USER 1000