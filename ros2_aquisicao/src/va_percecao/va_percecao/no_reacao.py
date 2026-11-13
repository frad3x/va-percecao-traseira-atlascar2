#!/usr/bin/env python3
"""
Nó 'reacao' — REAGE ao que a perceção traseira vai detetando.

SUBSCREVE
  percecao/seguranca   va_percecao_msgs/EstadoSeguranca  (do nó 'percecao')
  percecao/agentes     va_percecao_msgs/Agentes          (do nó 'percecao')
  <marcha_atras_topic> std_msgs/Bool    o carro está em marcha-atrás?   (opcional)
  <pisca_topic>        std_msgs/String  'esquerda' | 'direita' | ''     (opcional)
                       intenção do condutor de mudar de faixa

PUBLICA
  reacao/acao          va_percecao_msgs/Acao    nível (normal/atencao/perigo), comandos e motivo
  reacao/aviso         std_msgs/String          texto curto para um ecrã/voz (só quando muda)

Regras (cada comando fica ativo pelo menos 'retencao_s' segundos, para não piscar):
  - veículo atrás com TTC de risco        -> PERIGO  LIGAR_QUATRO_PISCAS
  - veículo atrás perto demais            -> ATENCAO AVISO_DISTANCIA_SEGURANCA
  - pisca para um lado ocupado            -> PERIGO  NAO_MUDAR_<LADO>
  - pisca para um lado com linha contínua -> ATENCAO NAO_MUDAR_<LADO>
  - veículo no ângulo morto               -> ATENCAO ATENCAO_ANGULO_MORTO
  - duas rodas a aproximar-se             -> ATENCAO ATENCAO_DUAS_RODAS
  - marcha-atrás com peão / obstáculo a < dist_parar  -> PERIGO  PARAR_MARCHA_ATRAS
  - marcha-atrás com obstáculo a < dist_abrandar      -> ATENCAO ABRANDAR_MARCHA_ATRAS
  - sem dados da perceção há > timeout_s  -> ATENCAO SEM_DADOS_CAMARA
"""
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from va_percecao_msgs.msg import Acao, Agentes, EstadoSeguranca

NIVEL = {"normal": 0, "atencao": 1, "perigo": 2}


class NoReacao(Node):
    def __init__(self):
        super().__init__("reacao")
        P = self.declare_parameter
        P("marcha_atras_topic", "/atlascar2/marcha_atras")
        P("pisca_topic", "/atlascar2/pisca")
        P("retencao_s", 1.0)
        P("timeout_s", 1.5)
        P("dist_parar", 1.0)        # m — marcha-atrás: parar
        P("dist_abrandar", 2.5)     # m — marcha-atrás: abrandar
        P("dist_peao", 4.0)         # m — marcha-atrás: peão no corredor
        g = lambda n: self.get_parameter(n).value
        self.retencao = float(g("retencao_s"))
        self.timeout = float(g("timeout_s"))
        self.d_parar, self.d_abrandar = float(g("dist_parar")), float(g("dist_abrandar"))
        self.d_peao = float(g("dist_peao"))

        self.marcha_atras = False
        self.pisca = ""
        self.ultimo_seg = None
        self.ultimos_agentes = None
        self.t_ultimo = 0.0
        self.ativos = {}            # comando -> (nivel, motivo, t_ultimo_visto)
        self._ultimo_texto = None

        self.create_subscription(EstadoSeguranca, "percecao/seguranca", self._on_seg, 10)
        self.create_subscription(Agentes, "percecao/agentes", self._on_agentes, 10)
        if g("marcha_atras_topic"):
            self.create_subscription(Bool, g("marcha_atras_topic"),
                                     lambda m: setattr(self, "marcha_atras", bool(m.data)), 10)
        if g("pisca_topic"):
            self.create_subscription(String, g("pisca_topic"),
                                     lambda m: setattr(self, "pisca", m.data.strip().lower()), 10)
        self.pub_acao = self.create_publisher(Acao, "reacao/acao", 10)
        self.pub_aviso = self.create_publisher(String, "reacao/aviso", 10)
        self.create_timer(0.25, self._vigiar)
        self.get_logger().info("reação pronta — à espera da perceção")

    # ------------------------------------------------------------------
    def _on_agentes(self, msg):
        self.ultimos_agentes = msg

    def _on_seg(self, msg):
        self.ultimo_seg = msg
        self.t_ultimo = time.monotonic()
        self._avaliar(msg)
        self._publicar(msg.header)

    def _vigiar(self):
        """Sem dados há demasiado tempo -> avisa (a câmara/perceção pode ter caído)."""
        if self.t_ultimo and time.monotonic() - self.t_ultimo > self.timeout:
            self._ativar("SEM_DADOS_CAMARA", "atencao", "perceção traseira sem dados")
            self._publicar(None)

    # ------------------------------------------------------------------
    def _ativar(self, comando, nivel, motivo):
        self.ativos[comando] = (nivel, motivo, time.monotonic())

    def _avaliar(self, s):
        self.ativos.pop("SEM_DADOS_CAMARA", None)
        # 1) veículo que segue atrás na nossa faixa
        if s.estado_distancia == "perigo":
            self._ativar("LIGAR_QUATRO_PISCAS", "perigo",
                         f"risco de colisão traseira (TTC {s.ttc_atras:.1f} s)")
        elif s.estado_distancia == "atencao":
            self._ativar("AVISO_DISTANCIA_SEGURANCA", "atencao",
                         f"veículo atrás a {s.distancia_atras:.0f} m")
        # 2) mudança de faixa: só reage se o condutor mostrar intenção (pisca)
        for lado, estado in (("esquerda", s.mudar_esquerda), ("direita", s.mudar_direita)):
            if self.pisca == lado and estado != "livre":
                nivel = "perigo" if estado == "ocupado" else "atencao"
                motivo = "faixa ocupada" if estado == "ocupado" else "linha contínua"
                self._ativar(f"NAO_MUDAR_{lado.upper()}", nivel, f"não mudar para a {lado}: {motivo}")
        # 3) avisos da perceção
        for a in s.avisos:
            t = a.texto.upper()
            if "ANGULO MORTO" in t:
                self._ativar("ATENCAO_ANGULO_MORTO", "atencao", a.texto.lower())
            elif "DUAS RODAS" in t:
                self._ativar("ATENCAO_DUAS_RODAS", "atencao", a.texto.lower())
        # 4) marcha-atrás / estacionamento
        if self.marcha_atras:
            livre = s.espaco_livre_atras
            peao = any(a.categoria == "peao" and math.isfinite(a.distancia) and a.distancia < self.d_peao
                       and a.faixa == "nossa" for a in (self.ultimos_agentes.agentes
                                                       if self.ultimos_agentes else []))
            peao = peao or any("PEAO" in a.texto.upper() for a in s.avisos)
            if peao or (math.isfinite(livre) and livre < self.d_parar):
                self._ativar("PARAR_MARCHA_ATRAS", "perigo",
                             "peão atrás do carro" if peao else f"obstáculo a {livre:.1f} m")
            elif math.isfinite(livre) and livre < self.d_abrandar:
                self._ativar("ABRANDAR_MARCHA_ATRAS", "atencao", f"obstáculo a {livre:.1f} m")

    def _publicar(self, header):
        agora = time.monotonic()
        for c in [c for c, (_, _, t) in self.ativos.items() if agora - t > self.retencao]:
            del self.ativos[c]
        acao = Acao()
        if header is not None:
            acao.header = header
        else:
            acao.header.stamp = self.get_clock().now().to_msg()
        ordem = sorted(self.ativos.items(), key=lambda kv: -NIVEL[kv[1][0]])
        acao.comandos = [c for c, _ in ordem]
        acao.nivel = ordem[0][1][0] if ordem else "normal"
        acao.motivo = ordem[0][1][1] if ordem else ""
        s = self.ultimo_seg
        acao.pode_mudar_esquerda = bool(s and s.mudar_esquerda == "livre")
        acao.pode_mudar_direita = bool(s and s.mudar_direita == "livre")
        acao.pode_fazer_marcha_atras = "PARAR_MARCHA_ATRAS" not in self.ativos
        self.pub_acao.publish(acao)

        texto = f"{acao.nivel.upper()}: {acao.motivo}" if acao.comandos else "NORMAL"
        chave = (acao.nivel, tuple(sorted(acao.comandos)))
        if chave != self._ultimo_texto:                 # só quando o nível/comandos mudam
            self._ultimo_texto = chave
            self.pub_aviso.publish(String(data=texto))
            log = {"normal": self.get_logger().info, "atencao": self.get_logger().warn,
                   "perigo": self.get_logger().error}[acao.nivel]
            log(f"{texto}  {acao.comandos}" if acao.comandos else texto)


def main(args=None):
    rclpy.init(args=args)
    no = NoReacao()
    try:
        rclpy.spin(no)
    except KeyboardInterrupt:
        pass
    finally:
        no.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
