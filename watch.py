import retro
import argparse

# Configuração do parser para receber argumentos da linha de comando
parser = argparse.ArgumentParser(description='Reproduzir um arquivo de replay .bk2')
parser.add_argument('--vid', type=str, required=True, help='Nome do arquivo .bk2 a ser reproduzido')

# Parseia os argumentos fornecidos pelo usuário
args = parser.parse_args()

# Exibe o nome do arquivo .bk2 fornecido
print(f"Reproduzindo o arquivo: {args.vid}")

# Carrega o arquivo de replay (.bk2)
movie = retro.Movie(args.vid)
movie.step()  # Avança para o primeiro frame do replay

# Configura o ambiente de jogo com base no jogo especificado no arquivo .bk2
env = retro.make(
    game=movie.get_game(),             # Obtém o nome do jogo do arquivo .bk2
    state=None,                        # Nenhum estado inicial específico
    use_restricted_actions=retro.Actions.ALL  # Usa todas as ações disponíveis
)

# Define o estado inicial da emulação com base no estado salvo no .bk2
env.initial_state = movie.get_state()
env.reset()  # Reinicia o ambiente

# Reproduz o replay frame a frame
while movie.step():  # Enquanto houver frames no arquivo
    keys = [
        movie.get_key(button, player)  # Obtém o estado do botão para cada jogador
        for player in range(movie.players)
        for button in range(env.num_buttons)
    ]
    # Passa os inputs registrados no .bk2 para o ambiente
    _obs, _rew, _done, _info = env.step(keys)
    # Renderiza o ambiente de jogo
    env.render()
