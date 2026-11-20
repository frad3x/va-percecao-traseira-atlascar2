# Marcacoes no pavimento detetadas numa imagem.
std_msgs/Header header
uint8 n_linhas            # 0, 1 ou 2 linhas da nossa faixa (aula 5, ex. 4b)
LinhaFaixa[] linhas
Marcacao[] marcacoes
float32 horizonte         # linha do horizonte estimada (pixel v)
