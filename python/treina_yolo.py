#!/usr/bin/env python3
"""
treina_yolo.py - Afinar (fine-tune) um YOLO com os TEUS dados anotados.

É aqui que o sistema "melhora com os dados": depois de anotares uma parte das
tuas imagens (caixas à volta de veículos/peões/sinais) em formato YOLO, este
script treina a partir de um modelo pré-treinado e produz um modelo melhor
(best.pt), que depois pões em MODELO no ../main.py.
(Para os SINAIS há um fluxo próprio e automático: ../treinar_sinais.py.)

Como anotar (exportar em formato YOLO):
  - CVAT, Roboflow ou labelImg.

Estrutura de dados esperada (formato Ultralytics):
  dados/treino/
    images/train/*.jpg   labels/train/*.txt
    images/val/*.jpg     labels/val/*.txt
    dados.yaml           (aponta para as pastas acima e lista as classes)

Exemplo de dados.yaml:
  path: ../dados/treino
  train: images/train
  val: images/val
  names: {0: veiculo, 1: peao, 2: duas_rodas, 3: sinal_stop, 4: sinal_limite, ...}

Uso:
  python treina_yolo.py --dados dados/treino/dados.yaml --epocas 50 --base yolo11n.pt
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados",  required=True, help="ficheiro dados.yaml")
    ap.add_argument("--base",   default="yolo11n.pt", help="modelo de partida")
    ap.add_argument("--epocas", type=int, default=50)
    ap.add_argument("--imgsz",  type=int, default=640)
    ap.add_argument("--nome",   default="atlascar_tras", help="nome da experiência")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        print("ERRO: instale o ultralytics -> pip install ultralytics", file=sys.stderr)
        print(f"(detalhe: {e})", file=sys.stderr)
        sys.exit(2)

    model = YOLO(args.base)                     # transfer learning a partir do pré-treinado
    model.train(data=args.dados, epochs=args.epocas, imgsz=args.imgsz,
                name=args.nome)
    # avaliação no conjunto de validação (precisão, recall, mAP)
    metrics = model.val()
    print("Resultados de validação:", metrics.results_dict if hasattr(metrics, "results_dict") else metrics)
    print("Modelo treinado guardado em runs/detect/%s/weights/best.pt" % args.nome)
    print("-> usa-o no main.py: MODELO = r\"runs/detect/%s/weights/best.pt\"" % args.nome)


if __name__ == "__main__":
    main()
