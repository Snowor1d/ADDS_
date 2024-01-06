from mesa import Model
from agent_renew import FightingAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import agent_renew
from agent_renew import WallAgent
import random

# goal_list = [[(80, 119), (79, 119), (78, 119), (77, 119)], #gate1 
#              [(198, 119), (197, 119), (196, 119)], #gate2
#              [(119, 19), (119, 20), (119, 21), (119, 22)]] #gate3 

# goal_list = [[(79, 119), (78, 119)], #gate1 
#              [(198, 119), (197, 119)], #gate2
#              [(119, 19), (119, 20)]] #gate3 

goal_list = [[0,50], [49, 50]]

    
            

def make_plane(xy1, xy2):
    new_plane = []
    
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]

    if(xy1[0]>xy2[0]):
        temp = x1
        x1 = x2
        x2 = temp
    if(xy1[1]>xy2[1]):
        temp = y1
        y1 = y2
        y2 = temp

    for i in range(x1, x2+1):
        for j in range(y1, y2+1):
            new_plane.append((i, j))

    return new_plane

class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, width: int, height: int):
        self.num_agents = number_agents
        self.grid = MultiGrid(width, height, False)
        self.headingding = ContinuousSpace(width, height, False, 0, 0)
        self.schedule = RandomActivation(self)
        self.schedule_e = RandomActivation(self)
        self.schedule_w = RandomActivation(self)
        self.running = (
            True  # required by the MESA Model Class to start and stop the simulation
        )

        self.datacollector_currents = DataCollector(
            {
                "Healthy Agents": FightingModel.current_healthy_agents,
                "Non Healthy Agents": FightingModel.current_non_healthy_agents,
            }
        )

        # Create agents
        # for i in range(self.num_agents):
        #     a = FightingAgent(i, self, self.random.randrange(4))
        #     self.schedule.add(a)

        #     # Add the agent to a random grid cell ## TODO: to make wall obstacle
            
        #     x = self.random.randrange(self.grid.width)
        #     y = self.random.randrange(self.grid.height)
        #     self.grid.place_agent(a, (x, y))

        # Create agents for test #두 aget가 서로 일직선 상으로 충돌하는 상황 
        # a = FightingAgent(0, self, [1,50], 0)
        # self.schedule.add(a)
        # self.grid.place_agent(a, (1, 50))

        # b = FightingAgent(1, self, [98, 48], 1)
        # self.schedule.add(b)
        # self.grid.place_agent(b, (98, 48))

        # b1 = FightingAgent(2, self, [98, 49], 1)
        # self.schedule.add(b1)
        # self.grid.place_agent(b1, (98, 49))

        # b2 =  FightingAgent(3, self, [98, 50], 1)
        # self.schedule.add(b2)
        # self.grid.place_agent(b2, (98, 50))

        # b3 = FightingAgent(4, self, [98, 51], 1)
        # self.schedule.add(b3)
        # self.grid.place_agent(b3, (98, 51))


        # b4 = FightingAgent(5, self, [98, 52], 1)
        # self.schedule.add(b4)
        # self.grid.place_agent(b4, (98, 52))


        # b5 = FightingAgent(6, self, [97, 48], 1)
        # self.schedule.add(b5)
        # self.grid.place_agent(b5, (97, 48))


        # b6 = FightingAgent(7, self, [97, 49], 1)
        # self.schedule.add(b6)
        # self.grid.place_agent(b6, (97, 49))


        # b7 = FightingAgent(8, self, [97, 50], 1)
        # self.schedule.add(b7)
        # self.grid.place_agent(b7, (97, 50))

        # b8 = FightingAgent(9, self, [97, 51], 1)
        # self.schedule.add(b8)
        # self.grid.place_agent(b8, (97, 51))

        # b9 = FightingAgent(10, self, [97, 52], 1)
        # self.schedule.add(b9)
        # self.grid.place_agent(b9, (97, 52))

        self.agent_place((40,20), (70, 80), 200)





        exit_rec = [] ## exit_rec list : exit_w * exit_h 크기 안에 (0,0)~(exit_w, exit_h) 토플 채워짐
        for i in range(0, agent_renew.exit_w):
            for j in range(0, agent_renew.exit_h):
                exit_rec.append((i,j))

        for i in range(len(exit_rec)): ## exit_rec 안에 agents 채워넣어서 출구 표현
            b = FightingAgent(i, self, [0,0], 10) ## exit_rec 채우는 agents의 type 10으로 설정;  agent_juna.set_agent_type_settings 에서 확인 ㄱㄴ
            self.schedule_e.add(b)
            self.grid.place_agent(b, exit_rec[i]) ##exit_rec 에 agents 채우기


        wall = [] ## wall list 에 (80, 200) ~ (80, 80), (80, 80)~(160, 80) 튜플 추가
        self.wall_matrix = list()
        for i in range(100):
            tmp = []
            for j in range(100):
                tmp.append(0)
            self.wall_matrix.append(tmp)
        
        from server_renew import NUMBER_OF_CELLS
        # for i in range(int(NUMBER_OF_CELLS*0.6)): ## 200*0.6=120
        #     wall.append((int(NUMBER_OF_CELLS*0.4),NUMBER_OF_CELLS-i-1)) ##(80, 200-i)
        # for i in range(int(NUMBER_OF_CELLS*0.4)): ## 200*0.4 = 80
        #     wall.append((int(NUMBER_OF_CELLS*0.4) + i, int(NUMBER_OF_CELLS*0.4))) ## (80+i, 80)
        # for i in range(int(NUMBER_OF_CELLS*0.7)): ## 200*0.7=140
        #     wall.append((int(NUMBER_OF_CELLS*0.3),NUMBER_OF_CELLS-i-1)) ##(60, 200-i)
        # for i in range(int(NUMBER_OF_CELLS*0.5)): ## 200*0.5 = 100
        #     wall.append((int(NUMBER_OF_CELLS*0.3) + i, int(NUMBER_OF_CELLS*0.3))) ## (60+i, 60)
        # wall = [] ## wall list 에 (80, 200) ~ (80, 80), (80, 80)~(160, 80) 튜플 추가
        # for i in range(120): ## 200*0.6=12
        #     wall.append((80,200-i-1)) ##(80, 200-i)
        # for i in range(80): ## 200*0.4 = 80
        #     wall.append((80+i, 80)) ## (80+i, 80)
            
        # map side wall
        for i in range(int(NUMBER_OF_CELLS)):
            wall.append((i, 0))
            wall.append((0, i))
            wall.append((i, int(NUMBER_OF_CELLS)-1))
            wall.append((int(NUMBER_OF_CELLS)-1, i))
           
            self.wall_matrix[i][0] = 1
            self.wall_matrix[0][i] = 1
            self.wall_matrix[i][int(NUMBER_OF_CELLS)-1] = 1
            self.wall_matrix[int(NUMBER_OF_CELLS)-1][0] = 1

        wall = wall + make_plane((40, 80), (40, 20))
        wall = wall + make_plane((40, 80), (70, 80))
        wall = wall + make_plane((70, 80), (70, 54))
        wall = wall + make_plane((70, 54), (90, 54))
        wall = wall + make_plane((90, 54), (90, 50))
        wall = wall + make_plane((90, 50), (70, 50))
        wall = wall + make_plane((70, 50), (70, 20))
        wall = wall + make_plane((40, 20), (70, 20))
        # wall.append((40, 50))
        # wall.append((40, 51))
        # wall.append((40, 49))
        # wall.append((40, 48))
        # wall.append((40, 52))
        # wall.append((40, 53))
        # wall.append((40, 47))
        # wall.append((40, 46))
        # wall.append((40, 54))
        # wall.append((40, 45))
        # wall.append((40, 55))
        

        # for i in range(int(80)):
        #     wall.append((i, 119))
        #     wall.append((79, int(NUMBER_OF_CELLS)-i-1))
        #     wall.append((119, int(NUMBER_OF_CELLS)-i-1))
        #     wall.append((119+i, 119))
        #     wall.append((119, 79-i))
        #     wall.append((119+i, 79))
           
        #     self.wall_matrix[i][119] = 1
        #     self.wall_matrix[79][int(NUMBER_OF_CELLS)-i-1] = 1
        #     self.wall_matrix[119][int(NUMBER_OF_CELLS)-i-1] = 1
        #     self.wall_matrix[119+i][119] = 1
        #     self.wall_matrix[119][79-i] = 1
        #     self.wall_matrix[119+i][79] = 1

        for i in goal_list:
            for j in i:
                if j in wall:    
                    wall.remove(j)
                    self.wall_matrix[j[0]][j[1]] = 0

        for i in range(len(wall)):
            c = FightingAgent(i, self, wall[i], 11)
            self.schedule_w.add(c)
            self.grid.place_agent(c, wall[i])
    

    def agent_place(self, xy1, xy2, num):
    
        agent_list = []
        x_len = xy2[0]-xy1[0]
        y_len = xy2[1]-xy1[1]

        check_list = [[0]*y_len for _ in range(x_len)]
        while(num>0):
            x = random.randint(xy1[0]+1, xy2[0]-1)
            y = random.randint(xy1[1]+1, xy2[1]-1)
            print(x, xy1[0])
            print(y, xy1[1])
            while(check_list[x-xy1[0]][y-xy1[1]] == 1):
                x = random.randint(xy1[0]+1, xy2[0]-1)
                y = random.randint(xy1[1]+1, xy2[1]-1)
            check_list[x-xy1[0]][y-xy1[1]] = 1
            num = num-1
            a = FightingAgent(num, self, [x,y], 0)
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))


    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.datacollector_currents.collect(self)  # passing the model

        # Checking if there is a champion
        if FightingModel.current_healthy_agents(self) == 0:
            self.running = False

    @staticmethod
    def current_healthy_agents(model) -> int:
        """Returns the total number of healthy agents.

        Args:
            model (SimulationModel): The model instance.

        Returns:
            (Integer): Number of Agents.
        """
        return sum([1 for agent in model.schedule.agents if agent.health > 0]) ### agent의 health가 0이어야 cureent_healthy_agents 수에 안 들어감
                                                                               ### agent.py 에서 exit area 도착했을 때 health를 0으로 바꿈

    @staticmethod
    def current_non_healthy_agents(model) -> int:
        """Returns the total number of non healthy agents.

        Args:
            model (SimulationModel): The model instance.

        Returns:
            (Integer): Number of Agents.
        """
        return sum([1 for agent in model.schedule.agents if agent.health == 0])
