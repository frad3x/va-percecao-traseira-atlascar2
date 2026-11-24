# Uma linha da faixa (aula 5: insertLaneBoundary / classifyLaneTypes).
string lado               # esquerda | direita (lado do CARRO) | divisoria (estimada, sem tinta) | outra
string tipo               # continua | tracejada | estimada
geometry_msgs/Point32[] pontos   # polilinha na imagem (x=u, y=v em pixeis)
# Tracejada: os tracos que EXISTEM na imagem (nunca um padrao inventado). Os pontos de todos
# os tracos vem seguidos em 'tracos'; tracos_n[k] = numero de pontos do traco k.
geometry_msgs/Point32[] tracos
uint16[] tracos_n
