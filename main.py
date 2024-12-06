import os
import neat
import neat.nn.recurrent
import retro
import numpy as np
import cv2
import pickle

env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)

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
        score, score_tracker = 0, 0
        coins, coins_tracker = 0, 0
        yoshi_coins, yoshi_coins_tracker = 0, 0
        pos_x_previous, pos_y_previous = 0, 0
        power_ups, power_ups_last = 0, 0
        end, jump = 0, 0

        done = False

        while not done:
            #env.render()
            frame += 1

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            img_array = ob.flatten() / 255.0
            nn_output = net.activate(img_array)

            ob, rew, done, info = env.step(nn_output)

            pos_x = info['x']
            score = info['score']
            coins = info['coins']
            end = info['endOfLevel']
            yoshi_coins = info['yoshiCoins']
            pos_y = info['y']
            jump = info['jump']
            power_ups = info['powerups']

            if pos_x > pos_x_max:
                fitness_current += 1
                pos_x_max = pos_x

            fitness_current += rew

            if pos_x > pos_x_previous:
                if jump > 0:
                    fitness_current += 10
                fitness_current += (pos_x / 10)
                pos_x_previous = pos_x
                counter = 0
            else:
                counter += 1
                fitness_current -= 0.1

            if pos_y < pos_y_previous:
                fitness_current += 10
                pos_y_previous = pos_y

            if power_ups < power_ups_last:
                fitness_current -= 50
                print("Lost Upgrade")
            elif power_ups > power_ups_last:
                fitness_current += 0.05
            power_ups_last = power_ups

            if score > score_tracker:
                fitness_current += score * 5
                score_tracker = score

            if coins > coins_tracker:
                fitness_current += coins * 5
                coins_tracker = coins

            if yoshi_coins > yoshi_coins_tracker:
                fitness_current += yoshi_coins * 5
                yoshi_coins_tracker = yoshi_coins

            if end == 1:
                fitness_current += 10000000
                done = True
            
            '''   
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            ''' 
            
            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


def main():
    print("Iniciando o treinamento...")

    config_path = os.path.join(os.path.dirname(__file__), "config-feedforward")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    try:
        pop = neat.Checkpointer.restore_checkpoint("neat-checkpoint")
        print("Checkpoint carregado com sucesso.")
    except FileNotFoundError:
        print("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(50))

    winner = pop.run(eval_genomes, 1000)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("Treinamento concluído. Melhor genoma salvo em 'winner.pkl'.")


if __name__ == "__main__":
    main()
