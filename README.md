# Classificação de Sementes usando OpenCV


### Extração de Características

**Características Globais**: descritores obtidos com base em todos os pontos de uma região, sua localização, intensidade e relações espaciais. (Área, perímetro e momentos)

**Características Locais**: calculadas a partir do contorno de um objeto ou a partir de pequenas regiões na imagem.
(Curvatura de contorno, detecção de bordas)

##### MOMENTOS DE HU  
Permite o cálculo da área, centroide ou permite IDENTIFICAR UM DETERMINADO OBJETO MESMO QUE TENHA SOFRIDO MUDANÇA de tamanho ou tenha sido rotacionado. Extrai características globais e invariantes com momentos invariantes à rotação, à translação e à escala.

##### LBP (LOCAL BINARY PATTERNS)
EXTRAI INFORMAÇÃO DE TEXTURA local. Rotula os pixels de uma imagem ao limitar a vizinhança de cada pixel e considera o resultado como um número binário. Basicamente faz um threshold e pode ser utilizado em conjunto com um histograma dessa imagem binarizada para identificar padrões em um conjunto de dados.

### Classificação

Para a classificação das sementes foram utilizados dois classificadores, sendo eles, **KMENS** e o **DBSCAN**.


## Para Compilar

>> python app.py
