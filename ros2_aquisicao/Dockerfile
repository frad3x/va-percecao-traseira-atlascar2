# Câmara traseira (Orbbec Astra Pro) do ATLASCAR2 — grupo G6
#
# UM Dockerfile, DUAS etapas (o compose escolhe com 'target'):
#   driver   — driver da câmara + gravação + URDF da câmara (leve)
#   percecao — a etapa 'driver' + o nosso pacote ROS 2 (nós 'percecao' e 'reacao')
#              e o núcleo de processamento em Python (YOLO/PyTorch — imagem grande)
# Assim o nó de processamento vive no MESMO contentor do driver (Parte 1, passo 5),
# e quem só precisar da câmara (Parte 2) usa a etapa 'driver' sem o PyTorch.
#
# Contexto do build = pasta do PROJETO (precisa de percecao/ e modelos/); o
# .dockerignore da raiz deixa entrar só o necessário. O compose já trata disto.
#
# Imagem base: a dos docentes, quando for publicada (BASE_IMAGE no .env).
ARG BASE_IMAGE=ros:jazzy-ros-base

# =============================== driver ===============================
FROM ${BASE_IMAGE} AS driver

# Toda a dependência vive aqui (regra do enunciado: se não está no Dockerfile,
# não existe). Driver UVC, compressão de imagem, gravação em MCAP, URDF
# (xacro + robot_state_publisher), visualização e utilitários de vídeo.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-usb-cam \
      ros-jazzy-image-transport-plugins \
      ros-jazzy-compressed-image-transport \
      ros-jazzy-image-view \
      ros-jazzy-rosbag2-storage-mcap \
      ros-jazzy-rviz2 \
      ros-jazzy-tf2-ros \
      ros-jazzy-xacro \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-foxglove-bridge \
      v4l-utils \
      procps \
      python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY ros2_aquisicao/launch/  launch/
COPY ros2_aquisicao/config/  config/
COPY ros2_aquisicao/urdf/    urdf/
COPY ros2_aquisicao/scripts/ scripts/
RUN chmod +x scripts/*.sh || true

# A imagem base já traz /ros_entrypoint.sh, que faz 'source' do ROS 2 e corre
# o comando dado no compose. Deixamos também o bash a fazer source.
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
CMD ["bash"]

# =============================== percecao ===============================
FROM driver AS percecao

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-vision-msgs \
      python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch: por omissão a versão CUDA 12.8 (funciona também sem GPU, em CPU).
# Imagem mais pequena, só CPU:  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision --index-url ${TORCH_INDEX}

# O núcleo do processamento (pacote Python 'percecao' — o mesmo da aplicação off-line).
# numpy<2: as mensagens ROS 2 do Jazzy (extensões C) foram compiladas com numpy 1.x.
# OpenCV "headless" (sem janelas) e <4.12, a última compatível com numpy 1.x.
WORKDIR /ws/projeto
COPY pyproject.toml README.md ./
COPY percecao/ percecao/
COPY modelos/ modelos/
RUN pip3 install --no-cache-dir --break-system-packages . "numpy<2" \
 && pip3 uninstall -y --break-system-packages opencv-python \
 && pip3 install --no-cache-dir --break-system-packages "opencv-python-headless<4.12" "numpy<2"

# Pacotes ROS 2 do grupo: va_percecao_msgs + va_percecao
WORKDIR /ws/ros
COPY ros2_aquisicao/src/ src/
RUN bash -c "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install"

# Pesos YOLO descarregados na 1.ª execução ficam aqui (volume no compose)
WORKDIR /ws/pesos
ENV VA_PROJETO=/ws/projeto
RUN echo "source /ws/ros/install/setup.bash" >> /root/.bashrc
COPY ros2_aquisicao/scripts/entrypoint_percecao.sh /entrypoint_percecao.sh
RUN chmod +x /entrypoint_percecao.sh
ENTRYPOINT ["/entrypoint_percecao.sh"]
CMD ["ros2", "launch", "va_percecao", "percecao.launch.py"]
