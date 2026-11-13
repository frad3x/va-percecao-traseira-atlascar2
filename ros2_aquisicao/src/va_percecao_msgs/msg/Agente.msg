# Um agente detetado pela camara traseira (um por caixa).
string VEICULO="veiculo"
string DUAS_RODAS="duas_rodas"
string PEAO="peao"
string SINAL="sinal"

string categoria          # veiculo | duas_rodas | peao | sinal
string classe             # carro, camiao, mota, bicicleta, peao, STOP, limite 50, ...
int32 id                  # ID do rastreio (ByteTrack); -1 se nao houver
float32 confianca

# caixa na imagem (pixeis): canto superior esquerdo + tamanho
float32 x
float32 y
float32 largura
float32 altura

# posicao no chao, no referencial da camara (x para tras do carro, y para a esquerda
# da IMAGEM), estimada pelo modelo monocular de chao plano. NaN se desconhecida.
float32 distancia         # m
float32 lateral           # m
string faixa              # nossa | esquerda | direita | fora   (lado do CARRO); "" se desconhecida
bool cortada              # a caixa sai por baixo da imagem (esta ainda mais perto)

float32 ttc               # tempo ate a colisao (s); +inf se nao se aproxima
float32 vel_aproximacao   # m/s (>0 = a aproximar-se); NaN se desconhecida
