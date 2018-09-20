import matplotlib.pyplot as plt

from envs.room import RoomEnv
from envs.room_spec import ROOM_PROBLEMS

FIGURES = {
    2: 'envs/vase.png'
}

if __name__ == '__main__':
    spec, current_state = ROOM_PROBLEMS['simple']
    env = RoomEnv(spec)
    fig, ax = plt.subplots()
    # Vase kept intact
    env.generate_and_plot_trajectory([2, 0, 0, 3], fig, ax)
    # Vase broken
    # env.generate_and_plot_trajectory([0, 0], fig, ax)
    plt.show()
