# Aquisição da câmara traseira — ATLASCAR2 (corre no portátil Ubuntu)

Este software **grava** os dados da câmara traseira (Orbbec Astra Pro) em
`rosbag2`/MCAP. Corre **no portátil Linux (Ubuntu 24.04)** com ROS 2 Jazzy, dentro
de Docker. Só precisas de ter o **Docker** instalado no portátil — o resto vive
dentro do contentor.

> O processamento (MATLAB/Python) é a outra parte do projeto, na pasta `../matlab`
> e `../python`. Aqui é **só gravar**. A pasta gravada é depois levada para o MATLAB.

---

## Preparação (uma vez)

1. **Docker** instalado no Ubuntu (docker + docker compose v2).
2. Ligar a câmara por USB e descobrir qual é:
   ```bash
   ls /dev/video*
   v4l2-ctl --list-devices
   ```
   Põe o dispositivo certo em `config/usb_cam.yaml` (`video_device`), p.ex.
   `/dev/video0` ou `/dev/video2`.
3. Copiar o ficheiro de configuração e preencher:
   ```bash
   cp .env.example .env
   ```
   No `.env` mete:
   - `CAM_Z` = **altura** da câmara ao solo (m) e `CAM_PITCH` = inclinação (rad;
     5° ≈ 0.087). Estes valores vão para a transformação do carro.
   - `DATA_DIR` **(opcional)** = onde guardar as gravações. Por omissão vão para
     a pasta **`./bags`** (aqui ao lado do código). Só defines `DATA_DIR` se
     quiseres gravar noutro sítio, p.ex. um SSD (`/media/teu_user/VA_SSD/data`).

---

## Gravar (no carro)

1. **Ligar a câmara** (constrói a imagem na 1.ª vez; pode demorar):
   ```bash
   docker compose up --build
   ```
   Deixa este terminal aberto — é a câmara a publicar.

2. **Ver a câmara / confirmar** (noutro terminal):
   ```bash
   ./scripts/ver.sh           # abre uma JANELA com o vídeo ao vivo (o mais simples)
   ./scripts/verificar.sh     # mostra ~15 Hz na imagem (confirma que chega)
   ./scripts/ver_rviz.sh      # alternativa com o rviz
   ```
   > Se a janela **não** abrir, a mensagem de erro aparece no próprio terminal
   > (o `ver.sh` corre em primeiro plano de propósito). Normalmente basta ter
   > feito `docker compose up --build` depois destas alterações.

3. **Gravar o percurso** (abre também uma **janela com o vídeo ao vivo**):
   ```bash
   ./scripts/gravar.sh percurso1
   ```
   Faz a condução. **Ctrl+C** para terminar a gravação (a janela fecha sozinha).
   A pasta gravada fica em **`./bags/`** (ou no `DATA_DIR` do `.env`) com um nome
   tipo `g02_camtras_20261104_153000_percurso1`.
   Para gravar **sem** a janela: `./scripts/gravar.sh percurso1 sem-janela`.
   > Se a janela do `gravar.sh` não abrir, grava na mesma e vê o vídeo noutro
   > terminal com `./scripts/ver.sh`.

   **Alternativa por compose** (sem script): com a câmara a correr,
   ```bash
   docker compose run --rm gravar
   # ou com outro nome:  PERCURSO=percurso2 docker compose run --rm gravar
   ```
   Ctrl+C para parar. (Esta via não abre a janela de vídeo; para ver usa o
   `./scripts/ver_rviz.sh`.)

   > Se adicionaste a janela agora, reconstrói a imagem uma vez antes:
   > `docker compose up --build` (a janela usa o `image_view`, novo no Dockerfile).

4. Repete o passo 3 para outros percursos (`percurso2`, etc.).

Quando acabares, Ctrl+C no terminal da câmara para desligar tudo.

---

## Nó de processamento (2.º serviço)

O `docker compose up` arranca **dois serviços** (como o enunciado pede): o
`camera` (driver) e o `processamento` (nó ROS 2). O nó `deteta_agentes` subscreve
a imagem, deteta **peões** (detetor HOG clássico do OpenCV, leve) e publica:
- `/atlascar2/camera_tras/num_agentes` — número de peões (std_msgs/Int32)
- `/atlascar2/camera_tras/agentes/image_raw/compressed` — imagem **anotada**

Para veres a imagem anotada ao vivo:
```bash
docker compose exec camera bash -lc "source /opt/ros/jazzy/setup.bash && \
  ros2 run image_view image_view --ros-args \
  -r image:=/atlascar2/camera_tras/agentes/image_raw/compressed \
  -p image_transport:=compressed"
```

> Isto é a versão **ao vivo mínima**. O processamento a sério (YOLO, linhas,
> sinais, trajetória) é feito **off-line em MATLAB** (`../matlab`), que é mais
> completo. Este nó existe para cumprir a arquitetura driver + processamento.

## O que é gravado

- `/atlascar2/camera_tras/image_raw/compressed` — a imagem (comprimida)
- `/atlascar2/camera_tras/camera_info` — parâmetros da câmara
- `/tf` e `/tf_static` — a posição da câmara no carro

**É a pasta gravada que levas para o MATLAB**: copia-a para
`../dados/bags/` no teu PC Windows e segue o `../matlab/README.md`.

---

## Calibração dos intrínsecos (opcional, melhora as linhas clássicas)

Sem calibração as imagens gravam na mesma; só os intrínsecos ficam inválidos.
Para calibrar, com a câmara a publicar, num terminal dentro do contentor
(`docker compose exec camera bash`):
```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 --square 0.025 \
  --ros-args -r image:=/atlascar2/camera_tras/image_raw \
             -r camera:=/atlascar2/camera_tras
```
Ajusta `--size` (nº de cantos internos do teu xadrez) e `--square` (lado em m).
No fim, copia os valores para `config/camera_info.yaml`.

> Alternativa sem calibrar: no MATLAB usa `METODO_LINHAS='rede'` (linhas por rede
> neuronal, não precisam de calibração).

---

## RGB-D: gravar também a PROFUNDIDADE (apoio ao estacionamento)

A gravação normal (acima) é só **cor** (`usb_cam`). Para o apoio ao estacionamento
precisas da **profundidade**, que na Astra Pro só sai pelo **driver Orbbec**. Isso
é um modo à parte (profile `rgbd`), com a sua própria imagem (`Dockerfile.rgbd`,
que compila o `OrbbecSDK_ROS2` — demora na 1.ª vez).

```bash
docker compose --profile rgbd build      # 1.ª vez (compila o driver Orbbec)
```
```bash
docker compose --profile rgbd up          # arranca a câmara RGB-D (cor + profundidade)
```
```bash
./scripts/gravar_rgbd.sh percurso1         # (noutro terminal) grava TUDO (cor+prof+nuvem+tf)
```

> **Confirmar no dispositivo real:** o pacote/launch do driver varia com a versão.
> Vê `ros2 pkg list | grep -i orbbec` dentro do contentor e, se preciso, ajusta
> `ORBBEC_PKG` / `ORBBEC_LAUNCH` no `.env` (ex. `astra.launch.py` → `astra_pro.launch.py`).
> Se a Astra Pro não for suportada pela versão do SDK, usa o driver que os docentes
> indicarem (troca em `Dockerfile.rgbd` / `launch/camera_tras_rgbd.launch.py`).

**Depois**, no PC do MATLAB: converte a profundidade do bag em nuvens de pontos e
analisa (ver `../matlab/README.md` → "Apoio ao estacionamento"):
```bash
python -m percecao.profundidade --fonte <pasta_bag> --saida_dir dados/nuvens
```
(corre a partir da pasta `Projeto/`)

---

## Tamanho dos ficheiros

As gravações ficaram mais pequenas sem mexer na arquitetura: em `config/usb_cam.yaml`
a câmara passou a **15 fps** (`framerate`) e o JPEG gravado tem **qualidade 40**
(`image_raw.compressed.jpeg_quality`). Juntos, dão ficheiros ~3–4× menores que
antes. Estes dois valores estão no `./config` (montado no contentor), por isso
basta **reiniciar** a câmara (`Ctrl+C` e `docker compose up` outra vez) — não é
preciso reconstruir. Queres ainda mais pequeno? baixa a qualidade (ex. 30) ou
os fps. Queres mais qualidade? sobe (ex. `framerate: 30.0`, `jpeg_quality: 80`).

---

## Notas e limitações

- **Modo normal só grava a cor** (`usb_cam`). Para a profundidade (RGB-D, apoio ao
  estacionamento) usa o **profile `rgbd`** com o driver Orbbec — ver a secção
  "RGB-D" acima. O modo de cor é o recomendado para a perceção 2D (linhas,
  veículos, peões, sinais); o RGB-D é só para o distintivo do estacionamento.
- Se não aparecer imagem: experimenta outro `/dev/videoX` e muda `pixel_format`
  para `yuyv` (talvez com `image_width: 640`, `image_height: 480`).
- `BASE_IMAGE` no `.env`: se os docentes derem uma imagem base, mete-a aí.
- Este código ainda **não foi testado com a câmara real** — os pontos a confirmar
  no carro são o `video_device` e o `pixel_format`.
