# Perceção — câmara traseira ATLASCAR2 (Tema 2) — versão Python

Perceção com a câmara traseira, **em Python**, em duas formas que partilham o mesmo
núcleo (`percecao/pipeline.py`):
- **Aplicação off-line** — vídeo, pasta de imagens ou rosbag → player, CSV e gráficos.
  Basta **dar Run no `main.py`** (ou `python -m percecao <fonte>`). Corre no Windows.
- **Pacote ROS 2 ao vivo** — o nó `percecao` subscreve a câmara e publica o que deteta;
  o nó `reacao` reage aos avisos. Ver "ROS 2 — perceção ao vivo" mais abaixo.

Deteta, numa só aplicação, com rastreio e estatísticas:
- **Linhas da faixa** (clássica, **sem calibração**), com a faixa preenchida no vídeo.
- **Veículos**, **peões/2 rodas**, **sinais** (YOLO com rastreio de IDs) + **sinais PT
  clássicos** por cor/forma (complementam o YOLO).
- **Tempo-para-colisão (TTC)** dos veículos que se aproximam por trás — o distintivo.
- **Apoio ao estacionamento (3D/RGB-D)** — chão por RANSAC + obstáculos + lugar livre.

## Como correr

```bash
python/.venv/Scripts/python.exe main.py        # Windows (usa o ambiente já preparado)
```
Ou ativa o ambiente e corre `python main.py`. No topo do `main.py` tens o bloco
**CONFIG** — muda `FONTE` para o teu vídeo/pasta/bag e dá Run:
```python
FONTE = r"dados/frames_bag.avi"   # .avi/.mp4, pasta de imagens, OU pasta/.mcap de um rosbag
```
Aceita **vídeo** (.avi/.mp4/…), **pasta de imagens**, ou **rosbag MCAP** (converte
sozinho para `dados/frames_bag.avi` e lê daí — sem ROS).

No fim abre o **player interativo**:
- **ESPAÇO** play/pausa · **a/d** (ou ←/→) recuar/avançar · **+/−** velocidade ·
  **barra** saltar para qualquer frame · **q** sair.

Saídas: `deteccoes.csv` (todas as caixas), `linhas.csv` (linhas e tipo por frame),
`seguranca.csv` (distâncias e avisos por frame) e gráficos (deteções/frame, TTC, distância).

**Linha de comandos** (a mesma aplicação, sem editar o código):
```bash
python -m percecao dados/bags/g02_camtras_20261104_160818_percurso1
python -m percecao video.mp4 --modelo yolo11n.pt --imgsz 640 --vel-kmh 50 --sem-graficos
python -m percecao --help
```
Instalada como pacote (`pip install ".[app]"`) fica também o comando `va-percecao`.

## Estrutura

| Ficheiro | Papel |
|---|---|
| `main.py` | Ponto de entrada da aplicação off-line — basta dar Run |
| `percecao/pipeline.py` | **Núcleo**: processa uma imagem (usado pela aplicação e pelo nó ROS) |
| `percecao/app.py`, `percecao/__main__.py` | A aplicação off-line e a sua linha de comandos |
| `percecao/seguranca.py` | Distância, faixa de cada agente, avisos (distância de segurança, mudança de faixa, ângulo morto, estacionamento) |
| `ros2_aquisicao/` | Aquisição (driver + gravação) e pacotes ROS 2 `va_percecao_msgs` / `va_percecao` |
| `percecao/fontes.py` | Vídeo / imagens / rosbag MCAP (com acesso aleatório p/ o player) |
| `percecao/detecao.py` | YOLO + rastreio (IDs) → veículos/peões/sinais |
| `percecao/linhas.py` | Linhas da faixa (método da aula 5: vista de cima + parábolas) |
| `ferramentas/descarregar_argoverse.py` | Imagens de câmaras traseiras (Argoverse 2) para testar |
| `percecao/sinais.py` | Sinais PT clássicos (cor HSV + forma) |
| `percecao/sinais_treinados.py` | Sinais com modelos treinados (deteção + 43 tipos) |
| `treinar_sinais.py`, `treino/` | Descarregar datasets + treinar os modelos de sinais |
| `percecao/aproximacao.py` | TTC dos veículos que seguem atrás |
| `percecao/desenho.py` | Overlay das deteções |
| `percecao/player.py` | Player interativo (OpenCV) |
| `percecao/exportar.py` / `avaliacao.py` | CSV / métricas vs ground truth |
| `percecao/percecao3d.py` | Perceção 3D (Open3D): chão + obstáculos + estacionamento |
| `percecao/profundidade.py` | Profundidade do bag → nuvem de pontos |

## Linhas da faixa (`percecao/linhas.py`) — método da aula 5
Segue o exemplo MATLAB *Visual Perception Using Monocular Camera* da aula 5, passo a
passo, mas **sem precisar de calibração**:

| Passo | MATLAB (aula 5) | Aqui |
|---|---|---|
| Vista de cima | `birdsEyeView` | IPM com chão plano; horizonte **estimado sozinho** (ponto de fuga) |
| Marcas | `segmentLaneMarkerRidge` (não Canny!) | ridge simétrico + contraste local (CLAHE) |
| Linhas | `findParabolicLaneBoundaries` | janelas deslizantes + **parábolas por RANSAC** (seguem curvas) |
| Rejeição | `validateBoundaryFcn`, `isOfMinLength`, força | curvatura no troço visível, comprimento mín., força ≥ 20% da maior |
| Tipo | `classifyLaneTypes` | **contínua / tracejada** pela cobertura da linha |
| Faixa do carro | `leftEgoBoundary` / `rightEgoBoundary` | linha mais próxima à esq. e à dir., com largura plausível |
| Dashboard (ex. 4b) | triângulos | **amarelo** se só há 1 linha, **vermelho** se nenhuma |
| Estatística (ex. 4a/4c) | estrutura por frame | `linhas.csv` (`frameID, frameTime, leftLine, leftLineType, …`) + gráfico |

No vídeo: esquerda a **vermelho**, direita a **verde** (como o `insertLaneBoundary`),
tracejada desenhada a tracejado, tipo escrito junto à linha, e uma **mini vista de cima**
no canto (como as janelas lado a lado da aula).

Detalhes que fazem a diferença (todos **medidos**, não adivinhados):
- **Horizonte automático**: é o único parâmetro crítico da vista de cima (com ele certo
  as linhas ficam paralelas). Estimado pela mediana do ponto de fuga ao longo do tempo.
- **Vista de cima relativa à imagem** (não em metros): sem calibração os metros só
  estariam certos se a altura da câmara estivesse certa. Só a estrada próxima entra
  (onde o chão é plano) — encostas e rochas ficavam como riscos falsos.
- **Brancas em `min(R,G,B)`**: o reflexo do sol no asfalto é quente (pouco azul) e
  desaparece; a tinta branca não. Amarelas num canal próprio `(R+G)/2 − B`.
- **Força ≥ 20%** (a aula usa 40% e diz que "pode ser ajustado"): ao lado de uma
  contínua forte, 40% eliminava sistematicamente as tracejadas.
- **Contínua/tracejada**: contínuas têm cobertura ≈ 1,00 e tracejadas ≈ 0,64 → limiar 0,88.

Resultados no vídeo inteiro (faixas com marcas):

| Vídeo | 2 linhas | Tipo certo (esq. / dir.) |
|---|---|---|
| Estrada de montanha com curvas | **99%** | 92% / **99%** |
| Autoestrada com sol de frente | **97%** | **98%** / **99,5%** |
| Dashcam 640×360 com cruzamento | 67% | — |

Parâmetros em `PARAMS` (topo de `linhas.py`); `horizonte` pode ser fixado se quiseres.

## Vídeos de câmara traseira para testar
- **Mixkit** (grátis, sem marca de água, licença Mixkit): os dois vídeos de teste em
  `dados/bags/vídeos teste/` vêm daqui — 52455 (autoestrada) e 52456 (montanha).
  Há mais em mixkit.co/free-stock-video/ (pesquisa "rear", "back window").
- **Argoverse 2** (investigação, CC BY-NC-SA 4.0, **sem conta**): câmaras traseiras
  `ring_rear_left/right`, ~320 imagens por gravação. Descarregar só essas câmaras:
  `python ferramentas/descarregar_argoverse.py --n 2` → dá o caminho para pôr em `FONTE`
  (16 downloads em paralelo: ~30 s por gravação). Já há 2 em `va_datasets/argoverse` (dentro do Projeto).
  ⚠️ Estas câmaras estão viradas **para trás e para o lado (~45°)** e apanham o carro
  em baixo: ótimas para **veículos/TTC**, fracas para **linhas** (a estrada atravessa a
  imagem na diagonal). Para linhas, usar `LINHAS_PARAMS = {"v_base": 0.82}` ou o nuScenes.
- **nuScenes** (CC BY-NC-SA 4.0, registo gratuito em nuscenes.org): câmara **`CAM_BACK`**
  (traseira centrada — a mais parecida com o ATLASCAR); o *mini* tem 10 cenas.
- **A2D2 / Audi** (CC BY-ND 4.0, AWS sem conta): câmara **`rearcenter`**; ficheiros grandes.

## Modelo YOLO e sinais pequenos
Por omissão `MODELO="yolo11m.pt"` com `IMGSZ=1280` (~28 fps com GPU). Testado no vídeo
de teste: deteta **16×** mais STOP que o `yolo11n` a 640 e dá mais confiança aos
veículos. **Sem GPU**: usa `yolo11n.pt` com `IMGSZ=640`.
Limite físico: sinais com ~10 px num vídeo 640×360 comprimido **não são detetáveis**
(nem ampliando — testado). Na câmara do ATLASCAR (1280×720) os sinais são maiores.
Para reconhecer **todos** os tipos de sinal (cedência, limites de velocidade…), o
caminho é treinar o YOLO num dataset de sinais com `python/treina_yolo.py`.

## Treinar o reconhecimento de sinais (43 tipos) — `treinar_sinais.py`
Basta dar Run (ou `python treinar_sinais.py`). Em duas fases, como nos sistemas reais:
1. **Detetor** — YOLO treinado no **GTSDB** (900 imagens com caixas): encontra os
   sinais e diz a categoria (proibição / perigo / obrigação / outros).
2. **Classificador** — YOLO-cls treinado no **GTSRB** (~39 mil recortes, 43 tipos):
   diz **qual** é o sinal (limite 50, cedência, STOP, rotunda, obras, passadeira…).

Ambos os datasets seguem a **convenção europeia de Viena** (a dos sinais portugueses).
O script descarrega-os das fontes oficiais (~1,9 GB) para `va_datasets/` (se existir no Projeto) ou `C:\Users\<user>\va_datasets`
— **fora do OneDrive**, para não sincronizar GB de imagens — converte-os, treina
(~1 h com GPU) e instala `modelos/sinais_det.pt` + `modelos/sinais_cls.pt`.
A partir daí o `main.py` usa-os **automaticamente**; sem eles usa o detetor clássico.

**Resultados do treino (RTX 5060, ~1 h):**
- Detetor (GTSDB, validação = 300 imagens oficiais de teste): **precisão 0.97 · recall 0.99 ·
  mAP50 0.991 · mAP50-95 0.890**.
- Classificador (GTSRB, 43 tipos, validação por faixas nunca vistas): **97,1%** top-1 (99,9% top-5).
- Visto em imagens: "limite 70" e "proibido ultrapassar (pesados)" certos; sem falsos
  positivos em placas de direção (que não fazem parte dos 43 tipos).
- No vídeo de teste **americano**: encontra os sinais e acerta o tipo quando há
  equivalente europeu (ex.: "DO NOT ENTER" → **sentido proibido**), falha nos que não
  têm (o losango amarelo dos EUA é lido como "estrada com prioridade", que é o seu
  significado na Europa). Para Portugal (convenção de Viena) é o domínio certo.

- `python treinar_sinais.py --so-preparar` → só descarrega/converte os dados.
- `python treinar_sinais.py --rapido` → teste de 1 época (não substitui os modelos).
- O GTSRB é dividido por **faixa** (cada sinal real tem ~30 frames quase iguais) para
  a validação não "ver" cópias do treino; o classificador **nunca** espelha imagens
  (senão "virar à esquerda" passava a "virar à direita").

## Sinais (YOLO + clássico)
O YOLO (COCO) só conhece `stop sign`; o detetor clássico apanha os restantes sinais
PT. Os limiares estão em `PARAMS` no topo de `percecao/sinais.py` — afinados para
**poucos falsos positivos** (interior). Numa estrada real, se perderes sinais, baixa
`sat_min`/`val_min` e sobe `area_rel_max`.

## Avaliação (métricas)
`main.py` exporta sempre `deteccoes.csv`. Para medir desempenho:
1. Anota algumas frames (ex. **CVAT**, **Roboflow** ou **labelImg**) sobre o vídeo.
2. Gera um `ground_truth.json`: lista `[ {"bb": [[x,y,w,h],...], "classe": ["veic","peoes","sinais",...]}, ... ]` (um por frame).
3. Põe o caminho em `FICH_GT` no `main.py` → imprime **TP/FP/FN, precisão, recall, F1, MAE** por grupo.

## Apoio ao estacionamento (3D / RGB-D)
Experimenta já (cena sintética):
```bash
python -m percecao.percecao3d        # desenha chão, obstáculos e a ROI (verde=livre)
```
Fluxo com dados reais (precisa da **profundidade**, modo `rgbd` em `ros2_aquisicao`):
1. Gravar com profundidade no Ubuntu: `docker compose --profile rgbd up` + `./scripts/gravar_rgbd.sh`.
2. Converter: `python -m percecao.profundidade --fonte <bag> --saida_dir dados/nuvens`.
3. Analisar: `python -c "from percecao import percecao3d; percecao3d.analisar('dados/nuvens/nuvem_00010.ply')"`.

## Ambiente
Python **3.12** + GPU NVIDIA. O ambiente está em `python/.venv` (torch cu128 com CUDA,
ultralytics, opencv, open3d, mcap). Reinstalar: ver `requirements.txt`. Mantém-se o
3.12 porque o 3.14 ainda **não** tem wheels de PyTorch-CUDA (RTX 5060/Blackwell) nem Open3D.

## Aquisição (gravar no carro)
Pasta `ros2_aquisicao/` (corre no PC Ubuntu, ROS 2 em Docker): cor (`docker compose up`)
e RGB-D (`--profile rgbd`). Ver `ros2_aquisicao/README.md`.

## ROS 2 — perceção ao vivo
Na mesma pasta, dois pacotes ROS 2 (Jazzy):
- `va_percecao_msgs` — mensagens (`Agentes`, `Pavimento`, `EstadoSeguranca`, `Acao`, …);
- `va_percecao` — nó **`percecao`** (subscreve `/atlascar2/camera_tras/image_raw/compressed`
  e publica agentes, pavimento, estado de segurança, marcadores 3D e imagem anotada) e nó
  **`reacao`** (subscreve esses avisos + pisca/marcha-atrás e publica a ação: quatro piscas,
  não mudar de faixa, parar a marcha-atrás, …).

```bash
cd ros2_aquisicao
docker compose build && docker compose up                 # câmara + perceção + reação
BAG=<pasta_do_bag> docker compose run --rm percecao_bag   # o mesmo sobre uma gravação
```
Tópicos, parâmetros e testes: `ros2_aquisicao/README.md` → "Perceção ao vivo (ROS 2)".
