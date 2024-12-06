import os
import neat
import neat.nn.recurrent
import retro
import numpy as np
import cv2
import pickle

env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)

def getRam(env):
    ram = []
    for k, v in env.data.memory.blocks.items():
        ram += list(v)
    return np.array(ram)

# Função para registrar logs de depuração
def log_debug(info):
    with open("debug_log.txt", "a") as log_file:
        log_file.write(info + "\n")

def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
        ob = env.reset()
        action = env.action_space.sample()
        
        inx, iny, inc = env.observation_space.shape
        inx = int(inx / 8)
        iny = int(iny / 8)
        
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness, fitness_current, frame, counter = 0, 0, 0, 0
        pos_x, pos_x_max = 0, 0
        img_array = []
        
        # Indica que o Mario está vivo
        done = False
        
        # Enquanto o Mario estiver vivo
        while not done:
            env.render()
            frame += 1
            
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            for x in ob:
                for y in x:
                    img_array.append(y)
                    
            img_array = np.array(img_array) / 255.0
            nn_output = net.activate(img_array)
            log_debug(f"Saída da rede neural: {nn_output}")
            
            ob, rew, done, info = env.step(nn_output)
            img_array = []
            
            fitness_current += rew
            
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                
            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)
                
            genome.fitness = fitness_current
        

# Função principal
def main():
    print("Iniciando o treinamento...")

    # Carrega o arquivo de configuração
    config_path = os.path.join(os.path.dirname(__file__), "config-feedforward")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Tenta carregar um checkpoint salvo, ou inicia um novo treinamento
    try:
        pop = neat.Checkpointer.restore_checkpoint("neat-checkpoint")
        print("Checkpoint carregado com sucesso.")
    except FileNotFoundError:
        print("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
        pop = neat.Population(config)

    # Configura relatórios
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(10))

    # Inicia o treinamento por 30 gerações
    winner = pop.run(eval_genomes, 30)

    # Salva o melhor genoma em um arquivo
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("Treinamento concluído. Melhor genoma salvo em 'winner.pkl'.")

# Executa o programa
if __name__ == "__main__":
    main()

    

