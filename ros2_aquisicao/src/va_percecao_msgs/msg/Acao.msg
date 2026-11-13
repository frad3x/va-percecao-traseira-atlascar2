# Reacao do sistema ao que a camara traseira ve (no 'reacao').
string NORMAL="normal"
string ATENCAO="atencao"
string PERIGO="perigo"
std_msgs/Header header
string nivel              # normal | atencao | perigo
string[] comandos         # ex.: LIGAR_QUATRO_PISCAS, NAO_MUDAR_ESQUERDA, PARAR_MARCHA_ATRAS
string motivo             # texto curto para o condutor
bool pode_mudar_esquerda
bool pode_mudar_direita
bool pode_fazer_marcha_atras
