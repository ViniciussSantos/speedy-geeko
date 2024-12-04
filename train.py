import os
import neat
import retro
import numpy as np
import pickle
import random
from utils import dec2bin, perform_action, actions_list
from rominfo import getInputs, getRam

RADIUS = 6
EXPECTED_INPUTS = (2 * RADIUS + 1) ** 2
STALL_COUNTER_LIMIT = 50

def log_debug(info):
    with open("debug_log.txt", "a") as log_file:
        log_file.write(info + "\n")

def eval_genome(genome, config):
    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    env = retro.make(game="SuperMarioWorld-Snes", state="YoshiIsland2", players=1)

    fitness_current = 0
    current_max_fitness = 0
    counter = 0
    stall_counter = 0
    previous_marioX = 0

    try:
        ob = env.reset()
        ram = getRam(env)
        done = False

        while not done:
            inputs, marioX, marioY = getInputs(ram, radius=RADIUS)

            if len(inputs) != EXPECTED_INPUTS:
                raise ValueError(f"Mismatch between input size ({len(inputs)}) and expected ({EXPECTED_INPUTS})")

            # Exploration-exploitation strategy
            if counter < 10:
                action = actions_list.index(66)  # Force initial action (run forward)
            else:
                nnOutput = net.activate(inputs)
                if random.random() < 0.1:  # Exploration
                    action = random.choice(range(len(actions_list)))
                else:  # Exploitation
                    action = np.argmax(nnOutput)

            reward, done, info = perform_action(action, env)

            # Penalize jumping without progress
            if marioX > previous_marioX:
                reward += 10
            else:
                if action == actions_list.index(130):
                    reward -= 5

            # Fitness update
            fitness_current += reward
            fitness_current += (marioX - previous_marioX) * 1.5
            previous_marioX = marioX

            # Debugging logs
            log_debug(f"Action chosen (index): {action}")
            log_debug(f"Action translated: {actions_list[action]}")
            log_debug(f"Reward obtained: {reward}")
            log_debug(f"Mario: X={marioX}, Y={marioY}")
            log_debug(f"Fitness: {fitness_current}")

            # Stall counter handling
            if stall_counter >= STALL_COUNTER_LIMIT:
                print("Mario is stuck! Resetting environment.")
                log_debug("Mario is stuck! Resetting environment.")
                env.reset()
                stall_counter = 0
                fitness_current -= 50  # Additional penalty

            # Update progress tracking
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if counter > 250:
                done = True

    except Exception as e:
        log_debug(f"Error during genome evaluation: {e}")
    finally:
        env.close()

    return max(fitness_current, 0)

def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def main():
    print("Starting training...")
    config_path = os.path.join(os.path.dirname(__file__), "config-feedforward.ini")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    try:
        pop = neat.Checkpointer.restore_checkpoint("neat-checkpoint")
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print("No checkpoint found. Starting fresh training.")
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(10))

    winner = pop.run(eval_genomes, 30)

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("Training completed. Best genome saved as 'winner.pkl'.")

if __name__ == "__main__":
    main()
