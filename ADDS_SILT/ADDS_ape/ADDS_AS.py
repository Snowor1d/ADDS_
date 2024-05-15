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
import time

import agent
import model
import time

#-------------------------#
visualization_mode = 'on' # choose your visualization mode 'on / off
run_iteration = 500
#-------------------------#
for j in range(run_iteration):
    print(j)
    if visualization_mode == 'off':
        s_model = model.FightingModel(5,50,50)
        if(run_iteration>0):
            del s_model
            s_model = model.FightingModel(5,50,50)


        n = 500  # n을 반복하려는 횟수로 설정
        #### 만약 n을 바꾼다면.. agent.py에 있는 robot_step 도 함께 바꿔주세요 ㅠㅠ####


        num_escaped_episodes = {
            "50%": 0,
            "80%": 0,
            "100%": 0
        }
        start_time = time.time()
        for i in range(n): # 에피소드 n번 돌린다
            s_model.step()
            print('에피소드 수',i+1)
            
            #print('남은 agent 수', num_remained_agent)
            print('num_remained_agent',agent.num_remained_agent)
        #-----------------------------------------------------------------------------------------------------------------------
            if i == 0: # 처음 생성된 agent 수 저장
                num_assigned_agent = agent.num_remained_agent

            if agent.num_remained_agent <= int(num_assigned_agent*0.5): # 50% 이상 빠져나가면 그때 에피소드 수 저장
                if num_escaped_episodes["50%"] == 0:
                    num_escaped_episodes["50%"] = i+1

            if agent.num_remained_agent <= int(num_assigned_agent*0.2): # 80% 이상 빠져나가면 그때 에피소드 수 저장
                if num_escaped_episodes["80%"] == 0:
                    num_escaped_episodes["80%"] = i+1
            
            if agent.num_remained_agent == 0: # 모두 빠져나가면 그때 에피소드 수 저장 , 텍스트 파일에 저장
                if num_escaped_episodes["100%"] == 0:
                    num_escaped_episodes["100%"] = i+1
                    print(num_escaped_episodes)
                    with open("example.txt", "a") as f:
                        f.write("{}, {}, {}\n".format(num_escaped_episodes["50%"], num_escaped_episodes["80%"], num_escaped_episodes["100%"]))

                break
            else:
                agent.num_remained_agent = 0 # 초기화
        #-----------------------------------------------------------------------------------------------------------------------

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
        ## grid size
        NUMBER_OF_CELLS = 50 ## square # 한 셀당 50cm x 50cm로 하겠음. 이 시뮬레이션 모델에서는 한 셀당 하나의 사람만 허용 cell 개수가 100개 -> 50m x 50m 크기의 맵
        SIZE_OF_CANVAS_IN_PIXELS_X = 1000
        SIZE_OF_CANVAS_IN_PIXELS_Y = 1000

        simulation_params = {
            "number_agents": NumberInput(
                "Hi, ADDS . Choose how many agents to include in the model", value=NUMBER_OF_CELLS
            ),
            "width": NUMBER_OF_CELLS,
            "height": NUMBER_OF_CELLS,
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
            
            if agent.type == 10: ## exit_rec 채우는 agent 
                portrayal = {
                    "Shape": "circle",
                    "Filled": "true",
                    "Color": "green", 
                    "r": 1,
                    "text": "",
                    "Layer": 0,
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
                    "Layer": 0,
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
                    portrayal["Color"] = "orange"
                portrayal["Layer"] = 1
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

        print("으악!!!!!!!!!!!!!")
        server = ModularServer(     # 이게 본체인데,,,
            FightingModel, # 내 모델
            #[grid, chart_healthy], # visualization elements 써줌
            [grid, chart_healthy],
            "ADDS crowd system", # 웹 페이지에 표시되는 이름
            simulation_params,
        )
        server.port = 8521  # The default
        print("으악!!!!!!!!!!!!!")
        server.launch()




 