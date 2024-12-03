import os

import neat
import retro
import pickle


def eval_genome(genome, config):
    raise Exception("not implemented")


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
