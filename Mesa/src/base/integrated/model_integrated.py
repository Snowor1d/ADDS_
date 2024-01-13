from mesa import Model
from agent_integrated import FightingAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import agent_integrated
from agent_integrated import WallAgent

# goal_list = [[(80, 119), (79, 119), (78, 119), (77, 119)], #gate1 
#              [(198, 119), (197, 119), (196, 119)], #gate2
#              [(119, 19), (119, 20), (119, 21), (119, 22)]] #gate3 

goal_list = [[(79, 119), (78, 119)], #gate1 
             [(198, 119), (197, 119)], #gate2
             [(119, 19), (119, 20)]] #gate3 

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
                "Remained Agents": FightingModel.current_healthy_agents, ## Healthy Agents -> Remained Agents
                "Non Healthy Agents": FightingModel.current_non_healthy_agents,
            }
        )

        # Create agents
        for i in range(self.num_agents):
            a = FightingAgent(i, self, self.random.randrange(4))
            self.schedule.add(a)

            # Add the agent to a random grid cell ## TODO: to make wall obstacle
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))


        exit_rec = [] ## exit_rec list : exit_w * exit_h 크기 안에 (0,0)~(exit_w, exit_h) 토플 채워짐
        for i in range(0, agent_integrated.exit_w):
            for j in range(0, agent_integrated.exit_h):
                exit_rec.append((i,j))

        for i in range(len(exit_rec)): ## exit_rec 안에 agents 채워넣어서 출구 표현
            b = FightingAgent(i, self, 10) ## exit_rec 채우는 agents의 type 10으로 설정;  agent_juna.set_agent_type_settings 에서 확인 ㄱㄴ
            self.schedule_e.add(b)
            self.grid.place_agent(b, exit_rec[i]) ##exit_rec 에 agents 채우기


        wall = [] ## wall list 에 (80, 200) ~ (80, 80), (80, 80)~(160, 80) 튜플 추가
        self.wall_matrix = list()
        for i in range(200):
            tmp = []
            for j in range(200):
                tmp.append(0)
            self.wall_matrix.append(tmp)
        
        from server_integrated import NUMBER_OF_CELLS
        # for i in range(int(NUMBER_OF_CELLS*0.6)): ## 200*0.6=120
        #     wall.append((int(NUMBER_OF_CELLS*0.4),NUMBER_OF_CELLS-i-1)) ##(80, 200-i)
        # for i in range(int(NUMBER_OF_CELLS*0.4)): ## 200*0.4 = 80
        #     wall.append((int(NUMBER_OF_CELLS*0.4) + i, int(NUMBER_OF_CELLS*0.4))) ## (80+i, 80)
        # for i in range(int(NUMBER_OF_CELLS*0.7)): ## 200*0.7=140
        #     wall.append((int(NUMBER_OF_CELLS*0.3),NUMBER_OF_CELLS-i-1)) ##(60, 200-i)
        # for i in range(int(NUMBER_OF_CELLS*0.5)): ## 200*0.5 = 100
        #     wall.append((int(NUMBER_OF_CELLS*0.3) + i, int(NUMBER_OF_CELLS*0.3))) ## (60+i, 60)
        # wall = [] ## wall list 에 (80, 200) ~ (80, 80), (80, 80)~(160, 80) 튜플 추가
        # for i in range(120): ## 200*0.6=120
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

        for i in range(int(80)):
            wall.append((i, 119))
            wall.append((79, int(NUMBER_OF_CELLS)-i-1))
            wall.append((119, int(NUMBER_OF_CELLS)-i-1))
            wall.append((119+i, 119))
            wall.append((119, 79-i))
            wall.append((119+i, 79))
           
            self.wall_matrix[i][119] = 1
            self.wall_matrix[79][int(NUMBER_OF_CELLS)-i-1] = 1
            self.wall_matrix[119][int(NUMBER_OF_CELLS)-i-1] = 1
            self.wall_matrix[119+i][119] = 1
            self.wall_matrix[119][79-i] = 1
            self.wall_matrix[119+i][79] = 1

        for i in goal_list:
            for j in i:
                if j in wall:
                    wall.remove(j) ##??
                    self.wall_matrix[j[0]][j[1]] = 0

        for i in range(len(wall)):
            c = FightingAgent(i, self, 11)
            self.schedule_w.add(c)
            self.grid.place_agent(c, wall[i])


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
