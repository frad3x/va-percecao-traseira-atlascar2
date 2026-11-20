# Imagem do grupo G6 (ROS 2 Jazzy) — câmara traseira do ATLASCAR2.
# Quando a imagem base dos docentes estiver publicada no repositório comum, basta trocar o BASE_IMAGE:
#   docker compose build --build-arg BASE_IMAGE=<imagem-base-do-repositorio-comum>
ARG BASE_IMAGE=ros:jazzy-ros-base
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

# Tudo o que a app precisa é instalado aqui. Se não está no Dockerfile, não existe.
# Driver da câmara (usb_cam), compressão de imagem, gravação MCAP, URDF, visualização.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-usb-cam \
        ros-jazzy-image-transport-plugins \
        ros-jazzy-rosbag2-storage-mcap \
        ros-jazzy-foxglove-bridge \
        ros-jazzy-rviz2 \
        ros-jazzy-xacro \
        ros-jazzy-robot-state-publisher \
        ros-jazzy-tf2-ros \
        ros-jazzy-vision-msgs \
        v4l-utils \
        python3-numpy \
        python3-pytest \
        python3-pip libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# PyTorch: por omissão CUDA 12.8 (funciona também sem GPU, em CPU). Imagem mais pequena,
# só CPU (é o que o CI usa):  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision --index-url ${TORCH_INDEX}

# Núcleo do processamento (pacote Python 'percecao' — o mesmo da aplicação off-line).
# numpy<2: as mensagens ROS 2 do Jazzy foram compiladas com numpy 1.x.
# OpenCV "headless" (sem janelas) e <4.12, a última compatível com numpy 1.x.
WORKDIR /ws/projeto
COPY pyproject.toml README.md ./
COPY percecao/ percecao/
COPY modelos/ modelos/
RUN pip3 install --no-cache-dir --break-system-packages . "numpy<2" \
 && pip3 uninstall -y --break-system-packages opencv-python \
 && pip3 install --no-cache-dir --break-system-packages "opencv-python-headless<4.12" "numpy<2"
# Pesos YOLO base dentro da imagem: no carro pode não haver internet
RUN cd modelos && python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11m.pt')"
ENV VA_PROJETO=/ws/projeto

# Workspace ROS 2: o pacote comum (va_planeamento) e os do grupo (va_g06, va_percecao_msgs)
WORKDIR /ws
COPY src/ src/
RUN source /opt/ros/jazzy/setup.bash && colcon build

# Entrypoint: carrega o ROS 2 e o workspace antes de qualquer comando
RUN printf '%s\n' '#!/bin/bash' 'set -e' \
        'source /opt/ros/jazzy/setup.bash' 'source /ws/install/setup.bash' 'exec "$@"' \
        > /entrypoint.sh && chmod +x /entrypoint.sh \
    && echo 'source /ws/install/setup.bash' >> /root/.bashrc
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
