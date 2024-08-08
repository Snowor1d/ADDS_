from mesa import Model
from mesa import Agent
from agent import FightingAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import os 
import agent
from agent import WallAgent
import random
import copy
import math
import numpy as np
import os
import time

import agent
import model
import time
import sys

#-------------------------#
visualization_mode = 'off' # choose your visualization mode 'on / off
run_iteration = 1500
number_of_agents = 11 # agents 수

#-------------------------#

def objective_function(switch):
    velocity_a = 5
    velocity_b = 5
    score = 0
    if visualization_mode == 'off':
        for run_iter in range(3):
            print("run_iter:", run_iter)
            for model_num in range(5):
                print(f"{model_num}번째 모델")
                s_model = model.FightingModel(5,50,50,model_num)
                s_model_r = copy.deepcopy(s_model)     
                s_model_r.make_robot() ## 자리를 옮겨봤어
                s_model_r.make_agents()

                ran_num = random.randint(10000,20000)
                s_model.make_agents2()
                s_model.random_agent_distribute_outdoor2(number_of_agents,ran_num)
                
                # s_model_r.make_robot()
                # s_model_r.make_agents()
                s_model_r.random_agent_distribute_outdoor(number_of_agents,ran_num)

                if(run_iteration>0):
                    del s_model
                    del s_model_r
                    ran_num = random.randint(10000,20000)
                    s_model = model.FightingModel(5,50,50, model_num)
                    s_model_r = copy.deepcopy(s_model)
                    s_model_r.make_robot()
                    (s_model_r.return_robot()).change_value(velocity_a,velocity_b,switch)
                    (s_model_r.return_robot()).change_learning_state(0)
                    s_model_r.make_agents()
                    s_model.make_agents2()
                    s_model_r.random_agent_distribute_outdoor(number_of_agents,ran_num)
                    s_model.random_agent_distribute_outdoor2(number_of_agents,ran_num)

                n = run_iteration  # n을 반복하려는 횟수로 설정
                #### 만약 n을 바꾼다면.. agent.py에 있는 robot_step 도 함께 바꿔주세요 ㅠㅠ####


                num_escaped_episodes = {
                    "50%": 0,
                    "80%": 0,
                    "100%": 0
                }
                start_time = time.time()
                weight_update_flag = 0
                for i in range(n): # 에피소드 n번 돌린다

                    s_model_r.step()

                    content =""
                    #print('num_remained_agent_r',s_model_r.num_remained_agent)
                # robot 있는 모델의 agent 수 저장
                #-----------------------------------------------------------------------------------------------------------------------
                    if i == 0: # 처음 생성된 agent 수 저장
                        num_assigned_agent = s_model_r.num_remained_agent

                    if s_model_r.num_remained_agent <= int(num_assigned_agent*0.5): # 50% 이상 빠져나가면 그때 에피소드 수 저장
                        if num_escaped_episodes["50%"] == 0:
                            num_escaped_episodes["50%"] = i+1
                            score += (i+1)
                    if s_model_r.num_remained_agent <= int(num_assigned_agent*0.2): # 80% 이상 빠져나가면 그때 에피소드 수 저장
                        if num_escaped_episodes["80%"] == 0:
                            num_escaped_episodes["80%"] = i+1
                            score += (i+1)
                    if s_model_r.num_remained_agent == 0: # 모두 빠져나가면 그때 에피소드 수 저장 , 텍스트 파일에 저장
                        if num_escaped_episodes["100%"] == 0:
                            num_escaped_episodes["100%"] = i+1
                            print(num_escaped_episodes)
                            score += (i+1)
                            with open("robot.txt", "a") as f:
                                f.write("{}번째 학습, {}, {}, {}\n".format(i, num_escaped_episodes["50%"], num_escaped_episodes["80%"], num_escaped_episodes["100%"]))
                            with open("result.txt", "a") as f:
                                f.write(str((num_assigned_agent-s_model_r.num_remained_agent-(num_assigned_agent-s_model.num_remained_agent))/(i+1))+"\n")

                            with open("weight.txt", 'w') as file2:
                                try: 
                                    robot_agent = s_model_r.return_robot()
                                    if robot_agent is not None:
                                        new_lines = [str(robot_agent.w1) + '\n', str(robot_agent.w2) + '\n', str(robot_agent.w3)+'\n', str(robot_agent.w4)]
                                        file2.writelines(new_lines)
                                except:
                                    print("********************************")
                                    print("ERROR 147")
                                    print("********************************")
                                    continue
                            
                            if not (robot_agent == None):
                                new_lines = [str(robot_agent.w1) + '\n', str(robot_agent.w2) + '\n', str(robot_agent.w3)+'\n', str(robot_agent.w4)]
                            file2.close()
                        break
                
                    
                    #print('에피소드 수',i+1)

                    if i+1 == run_iteration :

                        if (num_escaped_episodes["50%"] == 0):
                            score += (run_iteration * 4)
                        elif (num_escaped_episodes["80%"] == 0):
                            score += (run_iteration * 2) 
                        elif (num_escaped_episodes["100%"] == 0):
                            score += run_iteration 

                        with open("weight.txt", 'w') as file2:
                            try: 
                                robot_agent = s_model_r.return_robot()
                                if robot_agent is not None:
                                    new_lines = [str(robot_agent.w1) + '\n', str(robot_agent.w2) + '\n', str(robot_agent.w3) +'\n', str(robot_agent.w4)]
                                    file2.writelines(new_lines)
                            except:
                                print("********************************")
                                print("ERROR 202")
                                print("********************************")                        
                                continue
                        with open("result.txt", "a") as f:
                                f.write(str((num_assigned_agent-s_model_r.num_remained_agent-(num_assigned_agent-s_model.num_remained_agent))/(i+1))+"\n")
                        
                        
                        file2.close()
                    
                    s_model.num_remained_agent = 0 # 초기화
                    s_model_r.num_remained_agent = 0 # 초기화
                    
                end_time = time.time()
                execution_time = end_time - start_time
                print("코드 실행 시간:", execution_time, "초")
    print("score:", score)

    return -score

from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds={'switch': (0, 1)},
    random_state = 13
)

optimizer.maximize(init_points=9, n_iter=25)
best_params = optimizer.max['params']
best_value = optimizer.max['target']

print("최적의 파라미터:", best_params)
print("최적의 목표 함수값:", best_value)





 