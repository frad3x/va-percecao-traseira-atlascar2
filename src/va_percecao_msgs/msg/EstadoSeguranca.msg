# Resumo de seguranca da retaguarda numa imagem.
std_msgs/Header header

# veiculo que segue atras na NOSSA faixa
string estado_distancia   # livre | ok | atencao | perigo
float32 distancia_atras   # m (NaN se livre)
float32 distancia_minima  # m: distancia de seguranca usada (regra dos 2 s ou fixa)
float32 ttc_atras         # s (+inf se nao se aproxima)
float32 vel_aproximacao   # m/s (NaN se desconhecida)

# mudanca de faixa (lado do CARRO): livre | ocupado | proibido
string mudar_esquerda
string mudar_direita

# estacionamento / marcha-atras
float32 espaco_livre_atras   # m ate ao 1.o obstaculo no corredor do carro (NaN = nada)

Aviso[] avisos
