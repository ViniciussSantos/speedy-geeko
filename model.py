import pickle

import cv2
import neat
import retro
import os


def get_input_dimensions():
    assert env.observation_space.shape is not None
    x, y, _ = env.observation_space.shape
    return int(x / 8), int(y / 8)


def generate_input_data(observale_env, imgarray: list, x: int, y: int):
    imgarray.clear()
    observale_env = cv2.resize(observale_env, (x, y))
    if len(observale_env.shape) == 3:
        observale_env = cv2.cvtColor(observale_env, cv2.COLOR_BGR2GRAY)

    for pixel_row in observale_env:
        imgarray.extend(pixel_row)

    return imgarray


def close_msg(ram):
    if ram[0x1426] != 0:
        close = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        [env.step(close) for _ in range(2)]


def score_points(points, max_points, current_fit):
    if points > max_points:
        current_fit += points
        max_points = points
    return max_points, current_fit


def score_x_pos(x, max_x, current_fit, count):
    if x > max_x:
        current_fit += x / 100
        max_x = x
        count = 0
        return max_x, current_fit, count

    count += 1
    current_fit -= 10
    return max_x, current_fit, count


def score_y_pos(y, max_y, current_fit):
    if y < max_y:
        current_fit += 10
        max_y = y
    return max_y, current_fit


def score_collected_coins(coins, max_coins, current_fit):
    if coins > 0 and coins > max_coins:
        current_fit += coins - max_coins
        max_coins = coins
    return max_coins, current_fit


def score_collected_yoshicoins(yoshicoins, max_yoshicoins, current_fit):
    if yoshicoins > 0 and yoshicoins > max_yoshicoins:
        current_fit += (yoshicoins - max_yoshicoins) * 10
        max_yoshicoins = yoshicoins
    return max_yoshicoins, current_fit


def score_powerups(powerups, last_powerup, current_fit):
    power_up_changes = {
        (0, 1): -100,  # Lost power-up
        (1, 0): 10,  # Gained power-up
        (1, 2): -100,  # Downgraded from higher power-up
        (2, 1): 20,  # Upgraded to max power-up
        (2, 0): 20,  # Jumped directly to max power-up
    }
    fitness_adjustment = power_up_changes.get((powerups, last_powerup), 0)
    current_fit += fitness_adjustment
    last_powerup = powerups
    return powerups, current_fit


def eval_genomes(genomes, config):
    def evaluate_single_genome(genome_id, genome, config):
        observation = env.reset()[0]
        net = neat.nn.RecurrentNetwork.create(genome, config)

        x, y = get_input_dimensions()
        fitness, counter = 0, 0
        state = {
            'max_points': 0,
            'max_xPos': 0,
            'max_yPos': 0,
            'max_coins': 0,
            'max_yoshicoins': 0,
            'last_powerup': 0,
            'checkpoint_reached': False,
        }

        done = False
        neural_network_input = []

        while not done:
            env.render()

            neural_network_input = generate_input_data(
                observation, neural_network_input, x, y
            )
            neural_network_output = net.activate(neural_network_input)

            observation, reward, done, _, _ = env.step(neural_network_output)
            ram = env.get_ram()

            close_msg(ram)
            x_pos = ram[0x95] * 256 + ram[0x94]
            y_pos = ram[0x00D3]
            coins = ram[0x0DBF]
            yoshi_coins = ram[0x1420]
            powerup = ram[0x0019]
            checkpoint = ram[0x13CE]
            dead = ram[0x0071]
            level_end = ram[0x13D6]
            points = reward / 10

            state['max_points'], fitness = score_points(
                points, state['max_points'], fitness
            )
            state['max_xPos'], fitness, counter = score_x_pos(
                x_pos, state['max_xPos'], fitness, counter
            )
            state['max_yPos'], fitness = score_y_pos(y_pos, state['max_yPos'], fitness)
            state['max_coins'], fitness = score_collected_coins(
                coins, state['max_coins'], fitness
            )
            state['max_yoshicoins'], fitness = score_collected_yoshicoins(
                yoshi_coins, state['max_yoshicoins'], fitness
            )
            state['last_powerup'], fitness = score_powerups(
                powerup, state['last_powerup'], fitness
            )

            if checkpoint == 1 and not state['checkpoint_reached']:
                state['checkpoint_reached'] = True
                fitness += 1000

            if dead == 9:
                fitness -= 100
                done = True

            if level_end < 80:
                fitness += 5000
                done = True

            if counter == 250:
                fitness -= 125
                done = True

        # Print genome performance
        print(genome_id, fitness)

        # Update genome's fitness
        genome.fitness = fitness

    # Evaluate each genome in the population
    for genome_id, genome in genomes:
        evaluate_single_genome(genome_id, genome, config)


def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    try:
        pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint')
        print('Checkpoint loaded.')
    except FileNotFoundError:
        print('No checkpoint found. Starting training from scratch.')
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(10))

    winner = pop.run(eval_genomes)

    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)

    print('Training done.')


if __name__ == '__main__':
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
    main()
