#!/usr/bin/env python
# utils.py
# Utility functions for NEAT-based agent-environment interaction
# Obs: Alguns métodos do projeto do Olivetti foram refatorados e reaproveitados

import numpy as np

# Lista de ações possíveis no ambiente
# Representam comandos no jogo (ex.: correr para direita, pular)
actions_list = [66, 130, 128, 131, 386]

def dec2bin(dec):
    """
    Converte um número decimal em uma lista representando seu valor binário.
    Útil para traduzir comandos numéricos para o formato aceito pelo ambiente Retro.

    Args:
        dec (int): Número decimal representando uma ação.

    Returns:
        list: Representação binária da ação.
    """
    binN = []
    while dec != 0:
        binN.append(dec % 2)
        dec //= 2
    return binN

def perform_action(action, env):
    """
    Executa uma ação no ambiente várias vezes para garantir que ela tenha efeito.
    A quantidade de repetições é ajustada com base na ação selecionada.

    Args:
        action (int): Ação a ser executada, definida em `actions_list`.
        env: Instância do ambiente Retro.

    Returns:
        tuple: Total de recompensas acumuladas, flag `done` indicando fim do episódio,
               e informações extras do ambiente (`info`).
    """
    reward = 0
    # Número de iterações para cada ação (otimizado para o jogo)
    actions_iters = {
        64: 8, 128: 8,  # Ações básicas como esquerda ou direita
        66: 4, 130: 4,  # Correr
        131: 8, 67: 8,  # Correr e pular
        386: 4, 322: 4  # Girar e correr
    }
    iterations = actions_iters.get(action, 1)  # Padrão: 1 iteração
    for _ in range(iterations):
        obs, rew, done, info = env.step(dec2bin(action))
        reward += rew
        if done:
            break
    return reward, done, info
