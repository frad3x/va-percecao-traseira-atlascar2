# Aquisição da câmara traseira (Orbbec Astra Pro) - ATLASCAR2
# Imagem base: ROS 2 Jazzy. Se os docentes derem uma imagem base própria,
# muda BASE_IMAGE no .env (ou no compose) para essa imagem.
ARG BASE_IMAGE=ros:jazzy-ros-base
FROM ${BASE_IMAGE}

# Toda a dependência vive aqui (regra do enunciado: se não está no Dockerfile,
# não existe). Driver da câmara (UVC genérico), compressão de imagem, gravação
# em MCAP, visualização e utilitários de vídeo.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-usb-cam \
      ros-jazzy-image-transport-plugins \
      ros-jazzy-compressed-image-transport \
      ros-jazzy-image-view \
      ros-jazzy-rosbag2-storage-mcap \
      ros-jazzy-rviz2 \
      ros-jazzy-tf2-ros \
      ros-jazzy-foxglove-bridge \
      v4l-utils \
      procps \
      python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY launch/  launch/
COPY config/  config/
COPY scripts/ scripts/
COPY nodes/   nodes/
RUN chmod +x scripts/*.sh || true

# A imagem base já traz /ros_entrypoint.sh, que faz 'source' do ROS 2 e corre
# o comando dado no compose. Deixamos também o bash a fazer source.
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
CMD ["bash"]
