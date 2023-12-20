from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import NumberInput

from model_juna import FightingModel
from mesa.visualization.modules import CanvasGrid, ChartModule

## grid size
NUMBER_OF_CELLS = 200 ## square
SIZE_OF_CANVAS_IN_PIXELS_X = 1000
SIZE_OF_CANVAS_IN_PIXELS_Y = 1000

simulation_params = {
    "number_agents": NumberInput(
        "Hi, Juna(B). Choose how many agents to include in the model", value=NUMBER_OF_CELLS
    ),
    "width": NUMBER_OF_CELLS,
    "height": NUMBER_OF_CELLS,
}

# def schelling_draw(agent): ### 탈출 사각형..을 만들고 싶었는디 어디에 생기는지는 안 쓰나?
#     if agent is None:
#         return
#     portrayal = {"shape":"rect", "w":"agent_juna.exit_w", "h": "agent_juna.exit_h", "Filled":"true", "Layer":0}
#     portrayal["Color"] = "Red"
#     return portrayal

# canvas_element = CanvasGrid(schelling_draw, agent_juna.exit_w, agent_juna.exit_h, 500, 500)



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
            "Color": "blue", 
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
    portrayal["Color"] = "green"
    portrayal["Layer"] = 1
    return portrayal


'''
    # if the agent is alive we set its radius according to the its type
    if agent.type == 0:
        portrayal["r"] = 0.2

    elif agent.type == 1:
        portrayal["r"] = 0.4

    elif agent.type == 2:
        portrayal["r"] = 0.6

    elif agent.type == 3:
        portrayal["r"] = 0.9

    # according to the level o health of the agent we change the color of it
    if agent.health > 50:
        portrayal["Color"] = "green"
        portrayal["Layer"] = 1

    else:
        portrayal["Color"] = "red"
        portrayal["Layer"] = 2

    return portrayal
'''

grid = CanvasGrid(
    agent_portrayal,
    NUMBER_OF_CELLS,
    NUMBER_OF_CELLS,
    SIZE_OF_CANVAS_IN_PIXELS_X,
    SIZE_OF_CANVAS_IN_PIXELS_Y,
)

chart_healthy = ChartModule(
    [
        {"Label": "Healthy Agents", "Color": "green"},
        #{"Label": "Non Healthy Agents", "Color": "red"}, ## 그래프 상에서 Non Healthy Agents 삭제
    ],
    canvas_height = 300,
    data_collector_name = "datacollector_currents",
)


server = ModularServer(
    FightingModel,
    [grid, chart_healthy],
    "ADDS crowd system",
    simulation_params,
)
server.port = 8521  # The default
server.launch()
