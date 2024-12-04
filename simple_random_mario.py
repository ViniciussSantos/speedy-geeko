import retro

# Configura o ambiente para o jogo Super Mario World no estágio Yoshi's Island 2
env = retro.make(game="SuperMarioWorld-Snes", state="YoshiIsland2", players=1)

# Reinicia o ambiente para o estado inicial
env.reset()

# Loop para interação com o jogo
while True:
    # Realiza ações aleatórias e avança o jogo em um frame
    obs, rew, done, info = env.step(env.action_space.sample())
    # Renderiza o jogo na tela
    env.render()
    # Verifica se o jogo terminou e sai do loop
    if done:
        break
