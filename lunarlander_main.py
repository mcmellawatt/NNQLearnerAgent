import gym
import nn_qlearner

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    env.monitor.start('tmp/lunarlanderresults/', force=True, video_callable=False)
    agent = nn_qlearner.NNQLearnerAgent(env.action_space.n)
    reward = 0.0
    done = False

    for episode in range(1, 30000):
        reward, obs, done = 0.0, env.reset(), False
        action = agent.choose_action(obs, reward, done, episode)

        while not done:
            obs, reward, done, info = env.step(action)
            action = agent.choose_action(obs, reward, done, episode)

    env.close()

    env.monitor.close()



