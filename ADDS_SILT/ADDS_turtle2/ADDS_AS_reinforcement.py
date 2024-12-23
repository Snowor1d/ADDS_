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
MAP_SIZE = 50
#-------------------------#
model_num = random.randint(0,4)
for j in range(run_iteration):

    print(f"{j} 번째 학습 ")
    if visualization_mode == 'off':
        s_model_r = model.FightingModel(5,MAP_SIZE, MAP_SIZE, model_num)
        
        s_model_r.make_robot() ## 자리를 옮겨봤어
        s_model_r.make_agents()

        ran_num = random.randint(10000,20000)

        s_model_r.random_agent_distribute_outdoor(number_of_agents,ran_num)

        if(run_iteration>0):
            del s_model_r
            ran_num = random.randint(10000,20000)
            s_model_r = model.FightingModel(5,70,70,model_num)
            s_model_r.make_robot()
            s_model_r.make_agents()
            s_model_r.random_agent_distribute_outdoor(number_of_agents,ran_num)

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
                    
                file2.close()
            
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
        s_model_r = model.FightingModel(5, MAP_SIZE, MAP_SIZE,model_num)  
        ran_num = random.randint(10000,20000)
        s_model_r.make_agents()
        
        s_model_r.make_robot()
        s_model_r.random_agent_distribute_outdoor(number_of_agents,ran_num)
        
        

        ## grid size
        NUMBER_OF_CELLS = MAP_SIZE 
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
            if agent.type == 0: #끌려가는 agent  
                portrayal["Color"] = "magenta"
                portrayal["Layer"] = 1
                return portrayal
            if agent.type == 1: #myway aget
                portrayal["Color"] = "blue"
                portrayal["Layer"] = 1
                return portrayal
            if agent.type == 2: #agent를 follow 하는 agent
                portrayal["Color"] = "cyan"
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
            SIZE_OF_CANVAS_IN_PIXELS_Y
        )

        chart_healthy = ChartModule(
            [
                {"Label": "Remained Agents", "Color": "blue"},
                #{"Label": "Non Healthy Agents", "Color": "red"}, ## 그래프 상에서 Non Healthy Agents 삭제
            ],
            canvas_height = 300,
            data_collector_name = "datacollector_currents",
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

        

        server2.port = 8522
        
        server2.launch()
        
        



 