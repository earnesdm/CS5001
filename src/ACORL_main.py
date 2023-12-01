import ACORL
import utils
import numpy as np


def evaluate_agent(env_name, agent, episodes=10):
    eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)

    avg_reward = 0
    for _ in range(episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.get_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= episodes

    print("---------------------------------------")
    print(f"Evaluation over {episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def train(env_name, replay_buffer, state_dim, action_dim, train_steps, eval_period):
    # initialize the ACORL agent
    agent = ACORL.ACORL(state_dim, action_dim)

    # load the data into the replay buffer
    replay_buffer.load('./buffers/Default_BreakoutNoFrameskip-v0_0')

    evaluations = []

    # initialize the imitation learning portion
    for _ in range(1000):
        agent.train_step(replay_buffer, both=False)

    # main train step
    for i in range(int(train_steps)):
        agent.train_step(replay_buffer, both=True)

        if i % eval_period == 0:
            print(f"Train Step: {i}")
            evaluations.append(evaluate_agent(env_name, agent))
    np.save(f"./results/ACORL_Evaluations", evaluations)


if __name__ == "__main__":
    # Atari Specific
    atari_preprocessing = {
        "frame_skip": 4,
        "frame_size": 84,
        "state_history": 4,
        "done_on_life_loss": False,
        "reward_clipping": True,
        "max_episode_timesteps": 27e3
    }

    env_name = "BreakoutNoFrameskip-v0"

    # create the environment
    _, _, state_dim, action_dim = utils.make_env(env_name, atari_preprocessing)

    # create the replay buffer
    replay_buffer = utils.ReplayBuffer(state_dim, True, atari_preprocessing, batch_size=32, buffer_size=1e6, device='cpu')

    train(env_name, replay_buffer, state_dim, action_dim, train_steps=1e6, eval_period=5e4)


