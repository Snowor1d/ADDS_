from mesa import Agent
import math

ATTACK_DAMAGE = 50
INITIAL_HEALTH = 100
HEALING_POTION = 20

STRATEGY = 1


def set_agent_type_settings(agent, type):
    """Updates the agent's instance variables according to its type.

    Args:
        agent (FightingAgent): The agent instance.
        type (int): The type of the agent.
    """
    if type == 1:
        agent.health = 2 * INITIAL_HEALTH ## 200
        agent.attack_damage = 2 * ATTACK_DAMAGE ## 100
    if type == 2:
        agent.health = math.ceil(INITIAL_HEALTH / 2) ## 50
        agent.attack_damage = math.ceil(ATTACK_DAMAGE / 2) ## 25
    if type == 3:
        agent.health = math.ceil(INITIAL_HEALTH / 4) ## 25
        agent.attack_damage = ATTACK_DAMAGE * 4 ## 80


class FightingAgent(Agent):
    """An agent that fights."""

    def __init__(self, unique_id, model, type): 
        super().__init__(unique_id, model)
        self.type = type
        self.health = INITIAL_HEALTH
        self.attack_damage = ATTACK_DAMAGE
        self.attacked = False
        self.dead = False
        self.dead_count = 0
        self.buried = False
        set_agent_type_settings(self, type)

    def __repr__(self) -> str:
        return f"{self.unique_id} -> {self.health}"

    def step(self) -> None:
        """Handles the step of the model dor each agent.
        Sets the flags of each agent during the simulation.
        """
        # buried agents do not move (Do they???? :))
        if self.buried:
            return

        # dead for too long it is buried not being displayed 
        if self.dead_count > 4:
            self.buried = True
            return

        # no health and not buried increment the count
        if self.dead and not self.buried:
            self.dead_count += 1
            return

        # when attacked needs one turn until be able to attack
        if self.attacked:
            self.attacked = False
            return

        self.move()

    def attackOrMove(self, cells_with_agents, possible_steps) -> None:
        """Decides if the user is going to attack or just move.
        Acts randomly.

        Args:
            cells_with_agents (list[FightingAgent]): The list of other agents nearby.
            possible_steps (list[Coordinates]): The list of available cell where to go.
        """
        should_attack = self.random.randint(0, 1) ## 50% 확률로 attack
        if should_attack:
            self.attack(cells_with_agents)
            return

        print("I chose to not attack!") ## 안 때렸을 때에는 안 때렸다고 말함
        new_position = self.random.choice(possible_steps) ## 다음 step에 이동할 위치 설정
        self.model.grid.move_agent(self, new_position) ## 그 위치로 이동

    def attack(self, cells_with_agents) -> None:
        """Handles the attack of the agent.
        Gets the list of cells with the agents the agent can attack.

        Args:
            cells_with_agents (list[FightingAgent]): The list of other agents nearby.
        """
        agentToAttack = self.random.choice(cells_with_agents) ## agent끼리 마주쳤을 때 맞을 애는 랜덤으로 고름
        agentToAttack.health -= self.attack_damage ## 랜덤으로 골라진 맞을 애 health에 attack_damage 줌
        agentToAttack.attacked = True ## 맞은 애 attacked 됐다~ 
        if agentToAttack.health <= 0: ## health 가 0보다 작으면 dead
            agentToAttack.dead = True
        print(f'I attacked! and health left is {agentToAttack.health}') ## 맞았을 때 맞았다 말하고 남은 health 량 표시

    def move(self) -> None:
        """Handles the movement behavior.
        Here the agent decides if it moves,
        drinks the heal potion,
        or attacks other agent."""

        should_take_potion = self.random.randint(0, 100)
        if should_take_potion == 1: ## 1/100 확률로 포션 먹음
            self.health += HEALING_POTION ## health 20 증가
            print(f'Drinking my potion! and my health left is {self.health}')
            return

        possible_steps = self.model.grid.get_neighborhood( ## 다음 step에서 갈 수 있는 곳은 이웃 grid
            self.pos, moore=True, include_center=False ##? 
        )

        cells_with_agents = []
        # looking for agents in the cells around the agent
        for cell in possible_steps:
            otherAgents = self.model.grid.get_cell_list_contents([cell])
            if len(otherAgents): ## 주변에 agent 있니?
                for agent in otherAgents: ## 안 죽은 agent 들 cells_with_agents에 추가
                    if not agent.dead:
                        cells_with_agents.append(agent)

        # if there is some agent on the neighborhood
        if len(cells_with_agents): ## 주변 agent 수 만큼
            if STRATEGY == 1: ## 언제 1 되냐???
                self.attackOrMove(cells_with_agents, possible_steps)
            else: ## 주변에 있는 애들 attack
                self.attack(cells_with_agents)
        else: ## 주변에 agent 없으면
            new_position = self.random.choice(possible_steps) ## 가능한 step으로 랜덤하게 이동할 위치 설정
            self.model.grid.move_agent(self, new_position) ## 그 위치로 이동
