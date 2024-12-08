import os
import pickle

import cv2
import neat
import retro


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


def watch(genome, config):
    observation = env.reset()[0]
    net = neat.nn.RecurrentNetwork.create(genome, config)

    done = False
    neural_network_input = []

    x, y = get_input_dimensions()

    while not done:
        env.render()

        neural_network_input = generate_input_data(
            observation, neural_network_input, x, y
        )
        neural_network_output = net.activate(neural_network_input)

        observation, _, done, _, _ = env.step(neural_network_output)

        ram = env.get_ram()
        close_msg(ram)

        level_end = ram[0x13D6]

        if level_end == 1:
            done = True


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

    with open('winner.pkl', 'rb') as input_file:
        winner = pickle.load(input_file)

    watch(winner, config)


if __name__ == '__main__':
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
    main()
