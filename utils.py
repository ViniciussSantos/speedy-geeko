#!/usr/bin/env python
# utils.py
# Utility functions for NEAT-based agent-environment interaction
# Obs: Alguns métodos do projeto do Olivetti foram refatorados e reaproveitados

import numpy as np

# Todas as possíveis ações
actions_map = {'noop':0, 'down':32, 'up':16, 'jump':1, 'spin':3, 
               'left':64, 'jumpleft':65, 'runleft':66, 'runjumpleft':67, 
               'right':128, 'jumpright':129, 'runright':130, 'runjumpright':131, 
               'spin':256, 'spinright':384, 'runspinright':386, 'spinleft':320, 'spinrunleft':322
               }

# Vamos usar apenas um subconjunto
actions_list = [66,130,128,131,386]

def dec2bin(dec):
    
    binN = []
    while dec != 0:
        binN.append(dec % 2)
        dec //= 2
    return binN


def perform_action(action, env):
    reward = 0
    actions_iters = {
        66: 4,   # Correr para a direita
        130: 4,  # Pular
        128: 8,  # Andar para a direita
        131: 8,  # Correr e pular
        386: 4   # Girar e correr
    }

    # Padrão de iterações é 1 se a ação não estiver no dicionário
    iterations = actions_iters.get(action, 1)

    # Realiza a ação no ambiente
    for _ in range(iterations):
        obs, rew, done, info = env.step(dec2bin(action))
        reward += rew
        if done:
            break

    return reward, done, info

