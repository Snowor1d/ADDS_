from mesa import Model
from agent_juna import FightingAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import agent_juna
from agent_juna import WallAgent

#goal_list = [[(198, 60), (199, 60), (197, 60), (196, 60), (195, 60), (194, 60)
#             ,(198, 59), (199, 59), (197, 59)], [(0,0), (0,1), (1,0), (1,1)]]


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
        for i in range(self.num_agents):
            a = FightingAgent(i, self, self.random.randrange(4))
            self.schedule.add(a)

            # Add the agent to a random grid cell ## TODO: to make wall obstacle
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))


        exit_rec = [] ## exit_rec list : exit_w * exit_h 크기 안에 (0,0)~(exit_w, exit_h) 토플 채워짐

        ## exit square of section 2
        for i in range(0, agent_juna.exit_w):
            for j in range(0, agent_juna.exit_h):
                exit_rec.append((i,j))

        # ## exit line of section 1
        # from server_juna import NUMBER_OF_CELLS
        # for i in range(int(NUMBER_OF_CELLS*0.2)):
        #     exit_rec.append((int(NUMBER_OF_CELLS*0.8)+i, int(NUMBER_OF_CELLS*0.3))) ## (200*0.8+i, 200*0.3)



        wall = [] ## wall list 에 (60, 200) ~ (60, 60), (60, 60)~(160, 60) 튜플 추가
        from server_juna import NUMBER_OF_CELLS

        # section division wall
        for i in range(int(NUMBER_OF_CELLS*0.7)): ## 200*0.7=140
            wall.append((int(NUMBER_OF_CELLS*0.3),NUMBER_OF_CELLS-i-1)) ##(60, 200-i)
        for i in range(int(NUMBER_OF_CELLS*0.5)): ## 200*0.5 = 100
            wall.append((int(NUMBER_OF_CELLS*0.3) + i, int(NUMBER_OF_CELLS*0.3))) ## (60+i, 60)
        
        # map side wall
        for i in range(int(NUMBER_OF_CELLS)):
            wall.append((i, 0))
            wall.append((0, i))
            wall.append((i, int(NUMBER_OF_CELLS)-1))
            wall.append((int(NUMBER_OF_CELLS)-1, i))



        # wall = [] ## wall list 에 (80, 200) ~ (80, 80), (80, 80)~(160, 80) 튜플 추가
        # for i in range(120): ## 200*0.6=120
        #     wall.append((80,200-i-1)) ##(80, 200-i)
        # for i in range(80): ## 200*0.4 = 80
        #     wall.append((80+i, 80)) ## (80+i, 80)
            

        

    
        for i in range(len(exit_rec)): ## exit_rec 안에 agents 채워넣어서 출구 표현
            b = FightingAgent(i, self, 10) ## exit_rec 채우는 agents의 type 10으로 설정;  agent_juna.set_agent_type_settings 에서 확인 ㄱㄴ
            self.schedule_e.add(b)
            self.grid.place_agent(b, exit_rec[i]) ##exit_rec 에 agents 채우기

        
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
