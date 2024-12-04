import os

import neat
import retro
import pickle


def eval_genome(genome, config):
    env = retro.make(game="SuperMarioWorld-Snes", state="YoshiIsland2", players=1)
    obs = env.reset()
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    max_distance = 0
    done = False
    frame_count = 0

    while not done:
        obs = obs.flatten() / 255.0  # Normalizar pixels entre 0 e 1
        action_values = net.activate(obs)
        
        # Escolher a ação baseada na saída da rede neural
        action_idx = np.argmax(action_values)   # Índice da ação mais forte
        action = actions_list[action_idx]       # Mapear para uma ação válida

        # Usar performAction para executar a ação
        reward = performAction(action, env)
        
        # Atualizar o fitness
        fitness += reward
        info = env.unwrapped.data.lookup        # Pega informações extras como "x" se suportado

        # Atualizar distância máxima
        if info and 'x' in info:
            distance = info['x']
            if distance > max_distance:
                max_distance = distance

        # Penalizar paradas
        frame_count += 1
        if frame_count > 500 and max_distance <= distance:
            break
        return reward total


def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def main():
    print("STARTING TRAINING")
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.ini")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)

    for stats_config in [
        neat.StatisticsReporter(),
        neat.StdOutReporter(True),
        neat.Checkpointer(),
    ]:
        pop.add_reporter(stats_config)

    model = pop.run(eval_genomes, config)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    env = retro.make(game="SuperMarioWorld-Snes", state="YoshiIsland2", players=1)
    main()
