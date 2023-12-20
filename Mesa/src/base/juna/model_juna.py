from mesa import Model
from agent_juna import FightingAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import agent_juna
from agent_juna import WallAgent

class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, width: int, height: int):
        self.num_agents = number_agents
        self.grid = MultiGrid(width, height, False)
        self.headingding = ContinuousSpace(width, height, False, 0, 0)
        self.schedule = RandomActivation(self)
        self.schedule_e = RandomActivation(self)
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
        for i in range(0, agent_juna.exit_w + 1):
            for j in range(0, agent_juna.exit_h + 1):
                exit_rec.append((i,j))

        for i in range(len(exit_rec)): ## exit_rec 안에 agents 채워넣어서 출구 표현
            b = FightingAgent(i, self, 10) ## exit_rec 채우는 agents의 type 10으로 설정;  agent_juna.set_agent_type_settings 에서 확인 ㄱㄴ
            self.schedule_e.add(b)
            self.grid.place_agent(b, exit_rec[i]) ##exit_rec 에 agents 채우기
            


        # print(wall)
        # for pos in wall:
        #     agent_type = 'wall' ### 이걸 이렇게 하지말고 agent color, shape ... 을 각각 해주는 형식으로 해봐야겠당
        #     agent = agent_juna.WallAgent(pos, self, agent_type)
        #     # self.grid.position_agent(agent, pos[0], pos[1])
        #     self.grid.place_agent(agent, pos)
        #     self.schedule.add(agent)

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
