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
visualization_mode = 'on' # choose your visualization mode 'on / off
run_iteration = 1500
number_of_agents = 11 # agents 수
#-------------------------#
for j in range(run_iteration):

    print(f"{j} 번째 학습 ")
    if visualization_mode == 'off':
        s_model = model.FightingModel(5,50,50)
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
            s_model = model.FightingModel(5,50,50)
            s_model_r = copy.deepcopy(s_model)
            s_model_r.make_robot()
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
            
            s_model.step()
            print('num_remained_agent',s_model.num_remained_agent)
        # robot 없는 모델의 agent 수 저장
        #-----------------------------------------------------------------------------------------------------------------------
            if i == 0: # 처음 생성된 agent 수 저장
                num_assigned_agent = s_model.num_remained_agent

            if s_model.num_remained_agent <= int(num_assigned_agent*0.5): # 50% 이상 빠져나가면 그때 에피소드 수 저장
                if num_escaped_episodes["50%"] == 0:
                    num_escaped_episodes["50%"] = i+1

            if s_model.num_remained_agent <= int(num_assigned_agent*0.2): # 80% 이상 빠져나가면 그때 에피소드 수 저장
                if num_escaped_episodes["80%"] == 0:
                    num_escaped_episodes["80%"] = i+1
            
            # if s_model.num_remained_agent == 0: # 모두 빠져나가면 그때 에피소드 수 저장 , 텍스트 파일에 저장
            #     if num_escaped_episodes["100%"] == 0:
            #         num_escaped_episodes["100%"] = i+1
            #         print(num_escaped_episodes)
            #         with open("norobot.txt", "a") as f:
            #             f.write("{}번째 학습, {}, {}, {}\n".format(j, num_escaped_episodes["50%"], num_escaped_episodes["80%"], num_escaped_episodes["100%"]))
            #         # with open("")
            #         # robot 이 없는 모델에서 모두 탈출했다고 해서

            #         # file2 = open("weight.txt", 'w')
            #         # try : 
            #         #     robot_agent = s_model_r.return_robot()
            #         #     new_lines = [str(robot_agent.w1) + '\n', str(robot_agent.w2) + '\n', str(robot_agent.w3)]
            #         # except :
            #         #     file2.close()
            #         #     continue
                        
            #         # if not (robot_agent == None):
            #         #     new_lines = [str(robot_agent.w1) + '\n', str(robot_agent.w2) + '\n', str(robot_agent.w3)]
            #         #     file2.writelines(new_lines)
            #         # file2.close()
            #     break
            
                
        #-----------------------------------------------------------------------------------------------------------------------


            s_model_r.step()

            content =""
            print('num_remained_agent_r',s_model_r.num_remained_agent)
        # robot 있는 모델의 agent 수 저장
        #-----------------------------------------------------------------------------------------------------------------------
            if i == 0: # 처음 생성된 agent 수 저장
                num_assigned_agent = s_model_r.num_remained_agent

            if s_model_r.num_remained_agent <= int(num_assigned_agent*0.5): # 50% 이상 빠져나가면 그때 에피소드 수 저장
                if num_escaped_episodes["50%"] == 0:
                    num_escaped_episodes["50%"] = i+1

            if s_model_r.num_remained_agent <= int(num_assigned_agent*0.2): # 80% 이상 빠져나가면 그때 에피소드 수 저장
                if num_escaped_episodes["80%"] == 0:
                    num_escaped_episodes["80%"] = i+1
            
            if s_model_r.num_remained_agent == 0: # 모두 빠져나가면 그때 에피소드 수 저장 , 텍스트 파일에 저장
                if num_escaped_episodes["100%"] == 0:
                    num_escaped_episodes["100%"] = i+1
                    print(num_escaped_episodes)
                    with open("robot.txt", "a") as f:
                        f.write("{}번째 학습, {}, {}, {}\n".format(j, num_escaped_episodes["50%"], num_escaped_episodes["80%"], num_escaped_episodes["100%"]))
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
        
            
            print('에피소드 수',i+1)
            # if i % 20 == 0:
            #     a = 1
            #     try:
            #         robot_agent = s_model_r.return_robot()
            #         reward = a * (s_model.reward_distance_real() - s_model_r.reward_distance_real())
            #         if(reward>20):
            #             reward = 0.3
            #         elif(reward>10):
            #             reward = 0.2
            #         elif (reward>5):
            #             reward = 0.1
            #         elif (reward>-5):
            #             reward = -0.1
            #         elif (reward>-10):
            #             reward = -0.2
            #         elif (reward>-20):
            #             reward = -0.4
            #         else :
            #             reward = -0.6

            #         (s_model_r.return_robot()).update_weight(reward)

            #         file3 = open("correlation.txt", "a")
            #         file3.write(f"{robot_agent.w1} {robot_agent.w2} {robot_agent.w3} {robot_agent.w4} {reward}\n")
            #         file3.close()
            
            #     except : 
            #         continue
            #     #     continue
                # if(reward<0):
                #     reward = -math.log2(-reward)
                # if(reward>0):
                #     reward = math.log2(reward)


            if i+1 == run_iteration :
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






    from mesa.visualization.ModularVisualization import ModularServer
    from mesa.visualization.UserParam import NumberInput
    from model import FightingModel
    from mesa.visualization.modules import CanvasGrid, ChartModule



    import asyncio
    import os
    import platform
    import webbrowser
    from typing import ClassVar

    import tornado.autoreload
    import tornado.escape
    import tornado.gen
    import tornado.ioloop
    import tornado.web
    import tornado.websocket

    from mesa_viz_tornado.UserParam import UserParam


    if visualization_mode == 'on':
        s_model = model.FightingModel(5,50,50)
        s_model_r = copy.deepcopy(s_model)     
        ran_num = random.randint(10000,20000)
        s_model.make_agents2()
        s_model.random_agent_distribute_outdoor2(number_of_agents,ran_num)
        
        s_model_r.make_robot()
        s_model_r.make_agents()
        s_model_r.random_agent_distribute_outdoor(number_of_agents,ran_num)
        
        

        ## grid size
        NUMBER_OF_CELLS = 50 ## square # 한 셀당 50cm x 50cm로 하겠음. 이 시뮬레이션 모델에서는 한 셀당 하나의 사람만 허용 cell 개수가 100개 -> 50m x 50m 크기의 맵
        SIZE_OF_CANVAS_IN_PIXELS_X = 1000
        SIZE_OF_CANVAS_IN_PIXELS_Y = 1000

        simulation_params = {
            "number_agents": NumberInput(
                "Hi, ADDS . Choose how many agents to include in the model", value=NUMBER_OF_CELLS
            ),
            "width": NUMBER_OF_CELLS,
            "height": NUMBER_OF_CELLS
        }

        def agent_portrayal(agent):
            # if the agent is buried we put it as white, not showing it.
            if agent.buried:
                portrayal = {
                    "Shape": "circle",
                    "Filled": "true", ## ?
                    "Color": "white", 
                    "r": 0.01,
                    "text": "",
                    "Layer": 0,
                    "text_color": "black",
                }
                return portrayal
            if agent.type == 20: ## for exit_way_rec 
                portrayal = {
                    "Shape": "circle",
                    "Filled": "true",
                    "Color": "lightblue", 
                    "r": 1,
                    "text": "",
                    "Layer": 0,
                    "text_color": "black",
                }
                return portrayal
            if agent.type == 10: ## exit_rec 채우는 agent 
                portrayal = {
                    "Shape": "circle",
                    "Filled": "true",
                    "Color": "green", 
                    "r": 1,
                    "text": "",
                    "Layer": 1,
                    "text_color": "black",
                }
                return portrayal
            
            if agent.type == 11: ## wall 채우는 agent 
                portrayal = {
                    "Shape": "circle",
                    "Filled": "true",
                    "Color": "black", 
                    "r": 1,
                    "text": "",
                    "Layer": 2,
                    "text_color": "black",
                }
                return portrayal
            if agent.type == 12: ## for space visualization 
                portrayal = {
                    "Shape": "circle",
                    "Filled": "true",
                    "Color": "lightgrey", 
                    "r": 1,
                    "text": "",
                    "Layer": 0,
                    "text_color": "black",
                }
                return portrayal

                
            
                


            # the default config is a circle
            portrayal = {
                "Shape": "circle",
                "Filled": "true",
                "r": 0.5,
                ##"text": f"{agent.health} Type: {agent.type}",
                "text_color": "black",
            }

            # if the agent is dead on the floor we change it to a black rect
            if agent.dead:
                portrayal["Shape"] = "rect"
                portrayal["w"] = 0.2
                portrayal["h"] = 0.2
                portrayal["Color"] = "black"
                portrayal["Layer"] = 1

                return portrayal
            
            portrayal["r"] = 1
            if agent.type == 1: #끌려가는 agent  
                portrayal["Color"] = "lightsalmon"
                portrayal["Layer"] = 1
                return portrayal
            if agent.type == 2: 
                portrayal["Color"] = "magenta"
                portrayal["Layer"] = 1
                return portrayal
            if agent.type == 3: #robot
                if agent.drag == 1: #끌고갈때
                    portrayal["Color"] = "red" #빨강!!!!!!!!!!!1
                else:
                    portrayal["Color"] = "purple"
                portrayal["Layer"] = 2
                return portrayal

            portrayal["Color"] = "blue"
            portrayal["Layer"] = 1
            return portrayal

        grid = CanvasGrid(
            agent_portrayal,
            NUMBER_OF_CELLS,
            NUMBER_OF_CELLS,
            SIZE_OF_CANVAS_IN_PIXELS_X,
            SIZE_OF_CANVAS_IN_PIXELS_Y,
        )

        chart_healthy = ChartModule(
            [
                {"Label": "Remained Agents", "Color": "blue"},
                #{"Label": "Non Healthy Agents", "Color": "red"}, ## 그래프 상에서 Non Healthy Agents 삭제
            ],
            canvas_height = 300,
            data_collector_name = "datacollector_currents",
        )


        server = ModularServer(     # 이게 본체인데,,,
            FightingModel, # 내 모델
            #[grid, chart_healthy], # visualization elements 써줌
            [grid, chart_healthy],
            "ADDS crowd system", # 웹 페이지에 표시되는 이름
            simulation_params,
            8521,
            s_model
        )
        server2 = ModularServer(     # 이게 본체인데,,,
            FightingModel, # 내 모델
            #[grid, chart_healthy], # visualization elements 써줌
            [grid, chart_healthy],
            "ADDS crowd system", # 웹 페이지에 표시되는 이름
            simulation_params,
            8522,
            s_model_r
        )

        

        server.port = 8521  # The default
        server2.port = 8522
        
        if(os.fork()==0):
            server.launch()
        else:
            server2.launch()
        
        



 