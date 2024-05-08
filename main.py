from GymMoreRedBalls import GymMoreRedBalls
import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time
import wandb
import gym

from memory import ReplayMemory
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state) :
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1  # action을 선택할 때마다 step에 +1처리
    if sample > eps_threshold:  # 정책 확률이 입실론보다 클 때, greedy하게 선택/ 아니면 exploration
        with torch.no_grad():  # forward pass 동안 gradient 계산 비활성화
            return policy_net(state.to('cpu')).max(1)[1].view(1, 1)  # DQN에 넘겨준 state 정보로 policy_net에서 가장 큰 action 확률 선택
            # policy_net으로 DQN 네트워크 호출
    else:  # 아니면 action 1,2,3,4 중에 랜덤하게 선택
        return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)  # batch_size 만큼 sampling 해옴

    batch = Transition(*zip(*transitions))
    # transition은 사전에 정의한 구조체 느낌(튜플인데 state, reward값을 필드로 가짐)
    # zip을 통해 Transition 객체에서, 동일한 인덱스를 가진 필드값을 그룹화 함. 이후 unpacking 하여 인자로 전달
    # 각 Transition 객체의 sate, action, reward별로 합치는 것..?
    actions = tuple((map(lambda a: torch.tensor([[a]], device='cpu'), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device='cpu'), batch.reward)))

    # non_final_mask는 에피소드가 아직 종료되지 않은 상태들을 처리한다.
    # 아직 도달하지 않은 state에 대해 학습을 보장하기 위해..?
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device='cpu', dtype=torch.uint8)
    # next_state가 종료되지 않은 상태만 filtering     #None은 에피소드가 끝났음을 나타냄
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cpu')

    state_batch = torch.cat(batch.state).to('cpu')  # 각 1x1 텐서를 하나로 합침
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # action_batch 중 실제 선택된(값: 1) value를 추출

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # 종료되지 않은 상태에 대해 target_network를 사용하여 각 상태의 최대 보상을 계산
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 또한 .detach()를 사용해서 target_net의 가중치에 그라디언트가 영향을 미치지 않도록 하기 위한 것,
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # 그라디언트 클래핑을 이용해 발산 방지
    optimizer.step()

def get_state(obs): #obs는 (높이, 너비, 채널) 형태 => (채널, 높이, 너비) 형태가 되도록
    # 이미지 데이터 추출
    image_data = obs[0]

    # 이미지 데이터를 PyTorch 텐서로 변환
    state = torch.tensor(image_data['image'], dtype=torch.float32)

    # PyTorch의 모델에 입력으로 사용할 수 있는 형태로 차원 변경
    # 이미지 데이터의 차원 순서를 (높이, 너비, 채널)에서 (채널, 높이, 너비)로 변경
    state = state.permute(2, 0, 1)

    # 배치 차원 추가
    state = state.unsqueeze(0)

    return state


def train(env, n_episodes, render = False) :
    for episodes in range(n_episodes):
        # action = env.action_space.sample()
        # observation, reward, done, truncated, info = env.step(action)
        # print(f"step={i}, action={action}, observation={observation}, reward={reward}, done={done}, info={info}")
        # print("env.instr.s_done:", env.instrs.s_done)

        obs = env.reset(seed=123)
        state = get_state(obs)
        total_reward = 0.0

        for i in count() :
            action = select_action(state)
            observation, reward, done, truncated, info = env.step(action)
            print("action:", action[0].item())  # 어떤 행동을 하는지
            print("obs:", observation)  # 관측한 observation 프린트
            print("env.instr.s_done:", env.instrs.s_done)
            #obs, reward, done, info = env.step(action)
            total_reward += reward

            print(f"step={i}, action={action[0].item()}, reward={reward}")
            print("episodes :", episodes)
            if not done :
                next_state = get_state(obs)
            else :
                next_state = None


            reward = torch.tensor([reward], device = device)

            # 리플레이 메모리에 경험 추가
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            obsevation = next_state

            if steps_done > INITIAL_MEMORY:  # 모델이 환경과 어느정도 상호작용 했을 시, 최적화 수행.
                optimize_model()

                if steps_done % TARGET_UPDATE == 0:  # 1000-step마다 target_net 업데이트
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                #wandb.log({"episode_reward": total_reward})  # Wandb에 에피소드 보상을 보고
                print("total_reward",total_reward)
                break
        #if episodes % 10 == 0:
        #    print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episodes, i, total_reward))
            # episode : x/y는 episode x당 y번의 action을 취한 것이라 볼 수 있다(해당 에피소드에서 수행된 step의 총수 : t).

def test(env, n_episodes, policy, render=True):
    for episode in range(n_episodes):
        obs = env.reset(seed=123)
        state = get_state(obs)
        total_reward = 0.0


        for i in count():
            action = policy(state.to('cpu')).max(1)[1].view(1,1)
            #observation, reward, done, truncated, info = env.step(action)
            observation, reward, done, truncated, info = env.step(action)
            #obs, reward, terminated, truncated, info
            total_reward += reward
            print("action:", action[0].item())  # 어떤 행동을 하는지
            print("obs:", observation)  # 관측한 observation 프린트
            print("env.instr.s_done:", env.instrs.s_done)
            # obs, reward, done, info = env.step(action)
            total_reward += reward
            print("testing..")
            print(f"step={i}, action={action[0].item()}, reward={reward}")
            #print("episodes : ", episodes)

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return


device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

# hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 800 #1000
RENDER = False
lr = 1e-4
INITIAL_MEMORY = 50   #10000  #sh edited previous 500
MEMORY_SIZE = 10 * INITIAL_MEMORY

# create networks
policy_net = DQN(n_actions=3).to('cpu')
target_net = DQN(n_actions=3).to('cpu')
target_net.load_state_dict(policy_net.state_dict())

# setup optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

steps_done = 0
#create env
# env = GymMoreRedBalls(room_size=10, render_mode='human')
env = GymMoreRedBalls(room_size=10, render_mode='human') # render_mode 를 human 으로 한 위와 같이하면 실제로 창에 어떻게 행동하는지가 디스플레이됨.
env.reset(seed=123)

memory = ReplayMemory(MEMORY_SIZE)

train(env, 1)

torch.save(policy_net, "dqn_redball")
policy_net = torch.load("dqn_redball")
print("Start Testing..")
test(env, 1, policy_net, render=False)

#for i in range(1000):
	#action = env.action_space.sample()
	#observation, reward, done, truncated, info = env.step(action)
	#print(f"step={i}, action={action}, observation={observation}, reward={reward}, done={done}, info={info}")
	#print("env.instr.s_done:", env.instrs.s_done)

	#if done:
	#	env.reset(seed=123)
	#	break


