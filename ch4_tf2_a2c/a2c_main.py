# A2C main
# coded by St.Watermelon

## 에이전트를 학습하고 결과를 도시하는 파일
# 필요한 패키지 임포트
from a2c_learn import A2Cagent
import gymnasium as gym
#import gym


def main():

    max_episode_num = 100   # 최대 에피소드 설정
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


def main3():
    env = gym.make("LunarLander-v2", render_mode="human")
    env.action_space.seed(42)

    observation, info = env.reset(seed=42)

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

import tensorflow as tf

def main4():
    # 사용 가능한 GPU 목록 확인
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"GPU 사용 가능: {len(gpus)}개 GPU가 감지되었습니다.")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
    else:
        print("GPU가 감지되지 않았습니다. CPU만 사용 중입니다.")

    # TensorFlow의 연산이 GPU를 사용하는지 확인
    print("TensorFlow가 사용하는 디바이스 확인:")
    tf.debugging.set_log_device_placement(True)

    # 테스트 연산
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    c = a + b
    print(c)

if __name__=="__main__":
    main()
