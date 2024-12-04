# A2C main
# coded by St.Watermelon

## 에이전트를 학습하고 결과를 도시하는 파일
# 필요한 패키지 임포트
from a2c_learn import A2Cagent
import gymnasium as gym
#import gym


def main():

    max_episode_num = 1000   # 최대 에피소드 설정
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
    agent = A2Cagent(env)   # A2C 에이전트 객체

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()



def main2():
    env = gym.make('CartPole-v1')
    for i_episode in range(20):
        observation, _ = env.reset()
        # for t in range(100):
        #     env.render()
        #     print(observation) 
        #     action = env.action_space.sample()
        #     observation, reward, done, info = env.step(action)
        #     if done: 
        #         print("Episode finished after {} timesteps".format(t+1))      
        #         break


if __name__=="__main__":
    main()
