# Aquisição da câmara traseira — ATLASCAR2 (corre no portátil Ubuntu)

Este software **grava** os dados da câmara traseira (Orbbec Astra Pro) em
`rosbag2`/MCAP. Corre **no portátil Linux (Ubuntu 24.04)** com ROS 2 Jazzy, dentro
de Docker. Só precisas de ter o **Docker** instalado no portátil — o resto vive
dentro do contentor.

> Esta pasta tem **dois contentores**: a **aquisição** (driver da câmara + gravação) e a
> **perceção ao vivo** (pacotes ROS 2 `va_percecao_msgs` e `va_percecao`, que subscrevem a
> imagem, publicam o que detetam e reagem — ver "Perceção ao vivo (ROS 2)" abaixo).
> A aplicação de processamento **off-line** (sobre a gravação, com player e CSV) está na
> pasta do projeto (`../main.py` ou `python -m percecao`) e corre também no Windows.

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
3. **(Opcional)** Copiar o ficheiro de configuração e preencher. Sem ele, o
   `docker compose up` funciona na mesma com os valores por omissão:
   ```bash
   cp .env.example .env
   ```
   No `.env` mete:
   - `CAM_Z` = **altura** da câmara ao solo (m) e `CAM_PITCH` = inclinação (rad;
     5° ≈ 0.087). Estes valores vão para o URDF da câmara (`urdf/`).
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

## Perceção ao vivo (ROS 2)

O `docker compose up` arranca a câmara **e** a perceção. Na **1.ª vez** constrói as
imagens (a da perceção traz o PyTorch — demora e ocupa vários GB):
```bash
docker compose build
docker compose up                                        # só CPU (yolo11n, 640)
docker compose -f compose.yaml -f compose.gpu.yaml up    # GPU NVIDIA (no .env: MODELO=yolo11m.pt, IMGSZ=1280)
RVIZ=true docker compose up                              # + rviz: imagem anotada e agentes em 3D
```

```
driver (usb_cam) --image_raw/compressed--> percecao --agentes/pavimento/seguranca--> reacao --> acao, aviso
                                              |                                        ^
                                              +--> imagem_anotada, marcadores (rviz)   +-- pisca, marcha_atras
```

### Pacotes (`src/`)
| Pacote | O que tem |
|---|---|
| `va_percecao_msgs` | mensagens `Agente(s)`, `LinhaFaixa`, `Marcacao`, `Pavimento`, `Aviso`, `EstadoSeguranca`, `Acao` |
| `va_percecao` | nós `percecao` e `reacao`; launch `sistema`, `percecao`, `bag`; `config/percecao.yaml`; rviz |

O nó `percecao` usa **o mesmo núcleo** da aplicação off-line (pacote Python
`percecao`, classe `Pipeline`): YOLO11 + ByteTrack, linhas e marcações, sinais
treinados, distância/TTC e avisos. Trata sempre a imagem **mais recente**: se o
modelo for mais lento que a câmara, salta imagens em vez de acumular atraso.

### Tópicos (namespace `/atlascar2/camera_tras`)
| Tópico | Tipo | |
|---|---|---|
| `image_raw/compressed` | `sensor_msgs/CompressedImage` | **subscreve** (vem do driver) |
| `/atlascar2/velocidade` | `std_msgs/Float32` (m/s) | **subscreve**, opcional → regra dos 2 s |
| `percecao/agentes` | `va_percecao_msgs/Agentes` | veículos, duas rodas, peões, sinais: ID, distância, faixa (do carro), TTC |
| `percecao/deteccoes` | `vision_msgs/Detection2DArray` | o mesmo em tipos normalizados (para a Parte 2) |
| `percecao/pavimento` | `va_percecao_msgs/Pavimento` | linhas da faixa + tipo, passadeiras, linhas de paragem |
| `percecao/seguranca` | `va_percecao_msgs/EstadoSeguranca` | distância/TTC ao veículo de trás, mudar de faixa, espaço livre, avisos |
| `percecao/marcadores` | `visualization_msgs/MarkerArray` | agentes em 3D (rviz) |
| `percecao/imagem_anotada/compressed` | `sensor_msgs/CompressedImage` | imagem com o overlay |
| `/atlascar2/pisca` | `std_msgs/String` | **subscreve** (`reacao`): `esquerda` / `direita` / `""` |
| `/atlascar2/marcha_atras` | `std_msgs/Bool` | **subscreve** (`reacao`) |
| `reacao/acao` | `va_percecao_msgs/Acao` | nível + comandos (`LIGAR_QUATRO_PISCAS`, `NAO_MUDAR_ESQUERDA`, `PARAR_MARCHA_ATRAS`, …) |
| `reacao/aviso` | `std_msgs/String` | texto para o condutor, só quando muda |

Ver e testar (noutro terminal: `docker compose exec percecao bash`):
```bash
ros2 topic echo /atlascar2/camera_tras/reacao/aviso
ros2 topic echo /atlascar2/camera_tras/percecao/seguranca
ros2 topic pub --once /atlascar2/pisca std_msgs/String "{data: esquerda}"    # simula o pisca
ros2 topic pub --once /atlascar2/marcha_atras std_msgs/Bool "{data: true}"   # simula a marcha-atrás
```
Os parâmetros (modelo, limiares, tópicos) estão em `src/va_percecao/config/percecao.yaml`.

### Sobre uma gravação (sem câmara)
O bag publica a imagem gravada e os nós reagem como se fosse ao vivo:
```bash
docker compose build percecao
BAG=g02_camtras_20261104_160818_percurso1 docker compose run --rm percecao_bag
```
(`BAG` é a pasta dentro de `DATA_DIR`, por omissão `./bags`.)

### Um Dockerfile, duas etapas
Tudo vem do `Dockerfile` desta pasta (regra do enunciado: nada se instala no Ubuntu
além do Docker). O serviço `camera` usa a etapa **`driver`** (leve: `usb_cam`, gravação,
URDF); o serviço `percecao` usa a etapa **`percecao`** = `driver` + o nosso pacote ROS 2 +
o núcleo Python com o YOLO. Quem só precisar da câmara (Parte 2) usa só a etapa `driver`.

### URDF da câmara
`urdf/camera_tras.urdf.xacro` é o **fragmento URDF** a contribuir para o URDF comum do carro
(macro `camera_tras`: `camera_tras_link` e `camera_tras_optical_frame`, presos ao
`base_link`). Enquanto o URDF completo não corre, o serviço `camera` publica-o sozinho
(`urdf/camera_tras_isolada.urdf.xacro` + `robot_state_publisher`) com a pose do `.env`.

## O que é gravado

- `/atlascar2/camera_tras/image_raw/compressed` — a imagem (comprimida)
- `/atlascar2/camera_tras/camera_info` — parâmetros da câmara
- `/tf` e `/tf_static` — a posição da câmara no carro (do URDF da câmara)

Para a analisar **off-line** (player, CSV, gráficos), copia a pasta para
`../dados/bags/` no PC e corre a aplicação: `python -m percecao dados/bags/<pasta>`
(ou mete-a no `FONTE` do `main.py`). Também a podes reproduzir com a perceção ao
vivo (ver "Sobre uma gravação" acima).

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

> A perceção não precisa desta calibração: as linhas e as distâncias usam um modelo
> de chão plano com o horizonte estimado automaticamente.

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

**Depois**, no PC do processamento: converte a profundidade do bag em nuvens de
pontos e analisa (ver `../README.md` → "Apoio ao estacionamento"):
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
