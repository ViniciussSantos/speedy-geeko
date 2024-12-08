# Super Mario World AI - Inteligência Artificial - 2024.Q3

## Projeto
O projeto em grupo consiste na implementação de um agente inteligente capaz de jogar a fase Yoshilsland2 do jogo Super Mario World de Super Nintendo utilizando um dos algoritmos apresentados no curso de Inteligência Artificial da UFABC.
Os algoritmos para execução do agente são escritos em Python e usam a biblioteca **NEAT** para a utilização de redes neurais artificiais, juntamente com a biblioteca **Retro Gym** para a execução do jogo.

![Yoshi's Island 2](https://mario.wiki.gallery/images/9/97/YoshisIsland2.png)


## Integrantes
- Vinicius de Miranda Galvão Monteiro - RA: 11202020668
- Rodrigo Oliveira Toscano de Melo - RA: 11202131335
- Felipe Luiz dos Santos Chamelete - RA: 11202021024
- Helena de Alcântara Gomes de Souza - RA: 11201812345
- Vinicius Santana Santos - RA: 11201811841
- Luccas Henrique Figueira Cortes - RA: 11202130811


## Execução
```bash
# dependendo do seu sistema, use python3 ao inves de python
pyenv install 3.8
pyenv shell 3.8
python -m venv .venv
source .venv/bin/activate
pip install neat-python==0.92 numpy==1.24.4 opencv-python==4.8.1.78 stable-retro==0.9.2
cp rom.sfc .venv/lib/python3.8/site-packages/retro/data/stable/SuperMarioWorld-Snes/
#treinar a rede neural 
python model.py 
#assistir a rede neural vencedora (em winner.pkl) jogar 
python watch.py
```
