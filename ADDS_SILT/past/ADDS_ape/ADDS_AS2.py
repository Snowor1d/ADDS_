


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
    "model_number" : 1
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


server = ModularServer(     # 이게 본체인데,,,
    FightingModel, # 내 모델
    #[grid, chart_healthy], # visualization elements 써줌
    [grid, chart_healthy],
    "ADDS crowd system", # 웹 페이지에 표시되는 이름
    simulation_params,
)
server.port = 8522  # The default
server.launch()




