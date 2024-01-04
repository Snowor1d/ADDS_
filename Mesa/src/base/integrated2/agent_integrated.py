from mesa import Agent
import math
import numpy as np

ATTACK_DAMAGE = 50
INITIAL_HEALTH = 100
HEALING_POTION = 20
exit_w = 5
exit_h = 5
exit_area = [[0,exit_w], [0, exit_h]]
STRATEGY = 1
random_disperse = 1

exit_area = [[0,exit_w], [0,exit_h]]


# goal_list = [[(80, 119), (79, 119), (78, 119), (77, 119)], #gate1
#              [(90, 120)], #gate2
#              [(198, 119), (197, 119), (196, 119)], #gate3
#              [(119, 90)], #gate4
#              [(119, 19), (119, 20), (119, 21), (119, 22)], #gate5 
#              [(0,0), (0,1), (1,0), (0,0)] ] #gate6 

goal_list = [[(79, 119), (78, 119)],
               [(90, 120)], 
             [(198, 119), (197, 119)],
             [(119, 90)],
             [(119, 19), (119, 20)],
             [(0,0), (0,1), (1,0), (0,0)] 
             ]

# goal_list = [[(198, 60), (199, 60), (197, 60), (196, 60), (195, 60), (194, 60)
#               ,(198, 59), (199, 59), (197, 59)], [(0,0), (0,1), (1,0), (1,1)]]
#goal_list = [[(0,0), (0,1)]]

def check_stage(pose):
    # stage_1 = [[60,200], [60,200]] #    60 < x범위 < 200
    #                                #    60 < y범위 < 200
    #                                # stage_2 는, else
    # if(pose[0]>stage_1[0][0] and pose[0]<stage_1[0][1] and pose[1]>stage_1[1][0] and pose[1]<stage_1[1][1]):
    #     return 0
    # else:
    #     return 1

    stage_1 = [[0, 80], [120, 200]]
    stage_2 = [[80, 120], [120, 200]]
    stage_3 = [[120, 200], [120, 200]]
    stage_4 = [[120, 200], [80, 120]]
    stage_5 = [[120, 200], [0, 80]]

    x = pose[0]
    y = pose[1]

    if(pose[0]>stage_1[0][0] and pose[0]<stage_1[0][1] and pose[1]>stage_1[1][0] and pose[1]<stage_1[1][1]):
        return 0
    elif(pose[0]>stage_2[0][0] and pose[0]<stage_2[0][1] and pose[1]>stage_2[1][0] and pose[1]<stage_2[1][1]):
        return 1
    elif(pose[0]>stage_3[0][0] and pose[0]<stage_3[0][1] and pose[1]>stage_3[1][0] and pose[1]<stage_3[1][1]):
        return 2
    elif(pose[0]>stage_4[0][0] and pose[0]<stage_4[0][1] and pose[1]>stage_4[1][0] and pose[1]<stage_4[1][1]):
        return 3
    elif(pose[0]>stage_5[0][0] and pose[0]<stage_5[0][1] and pose[1]>stage_5[1][0] and pose[1]<stage_5[1][1]):
        return 4
    else:
        return 5

def central_of_goal(goals):
    real_goal = [0, 0]
    for i in goals:
        real_goal[0] += i[0]
        real_goal[1] += i[1]
    real_goal[0] /= len(goals)
    real_goal[1] /= len(goals) 
    return real_goal

def check_departure(pose, goals):
    for i in goals:
        if (i[0]>pose[0] and i[1]>pose[1]):
            return True
    return False

 # goals의 가운데를 가져오는 함수
 # 어디로 향하게 할 것인가? -> goals의 가운데 

class WallAgent(Agent): ## wall .. 탈출구 범위 내에 agents를 채워넣어서 탈출구라는 것을 보여주고 싶었음.. 
    def __init__(self, pos, model, agent_type):
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type


        # wall = [] ## wall list : exit_w * exit_h 크기 안에 (0,0)~(exit_w, exit_h) 토플 채워짐
        # for i in range(0, exit_w + 1):
        #     for j in range(0, exit_h + 1):
        #         wall.append((i,j))
        # # print(wall)
        # for pos in wall:
        #     agent_type = 'wall'
        #     agent = WallAgent(pos, self, agent_type)
        #     self.grid.position_agent(agent, pos[0], pos[1])
        #     self.schedule.add(agent)



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
    if type == 10: ## 구분하려고 아무 숫자 함, exit_rec 채우는 agent type
        agent.health = 500 ## ''
        agent.attack_damage = 0 ## ''
    if type == 11: ## 마찬가지.. 이건 wall list 채우는 agent의 type
        agent.health = 500
        agent.attack_damage = 0

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
        self.which_goal = 0

        self.xy = [0, 0]
        self.vel = [0, 0]
        self.acc = [0, 0]
        self.mass = 3
        self.xy[0] = self.random.randrange(self.model.grid.width)
        self.xy[1] = self.random.randrange(self.model.grid.height)
        
        set_agent_type_settings(self, type)

    def __repr__(self) -> str:
        return f"{self.unique_id} -> {self.health}"

    def step(self) -> None:
        global exit_area
        global goal_list

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
        if (check_departure([self.xy[0],self.xy[1]], goal_list[len(goal_list)-1])):
            self.dead = True
            self.health = 0 ## 이게 0이어야 current healthy agent 수에 포함이 안 됨 ~!

        # if (self.which_goal != (len(goal_list)-1)):
        #     if(check_departure([self.xy[0], self.xy[1]], goal_list[self.which_goal])):
        #         self.which_goal += 1

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
        new_position = self.random.choice(possible_steps) ## 다음 step에 이동할 위치 설정
        self.model.grid.move_agent(self, new_position) ## 그 위치로 이동

    def attack(self, cells_with_agents) -> None:
        """Handles the attack of the agent.
        Gets the list of cells with the agents the agent can attack.

        Args:
            cells_with_agents (list[FightingAgent]): The list of other agents nearby.
        """
        agentToAttack = self.random.choice(cells_with_agents) ## agent끼리 마주쳤을 때 맞을 애는 랜덤으로 고름
        ##agentToAttack.health -= self.attack_damage ## 랜덤으로 골라진 맞을 애 health에 attack_damage 줌 ###인데 공격 못하게(damage 없도록) 바꿈.
        agentToAttack.attacked = True ## 맞은 애 attacked 됐다~ 
        if agentToAttack.health <= 0: ## health 가 0보다 작으면 dead
            agentToAttack.dead = True

    def move(self) -> None:
        global goal_list
        """Handles the movement behavior.
        Here the agent decides   if it moves,
        drinks the heal potion,
        or attacks other agent."""

        # should_take_potion = self.random.randint(0, 100)
        # if should_take_potion == 1: ## 1/100 확률로 포션 먹음
        #     self.health += HEALING_POTION ## health 20 증가
        #     print(f'Drinking my potion! and my health left is {self.health}')
        #     return

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
        # if len(cells_with_agents): ## 주변 agent 수 만큼
        #     if STRATEGY == 1: ## 언제 1 되냐???
        #         self.attackOrMove(cells_with_agents, possible_steps)
        #     else: ## 주변에 있는 애들 attack
        #         self.attack(cells_with_agents)
        new_position = possible_steps[0]
        # for i in possible_steps:
        #     distance_to_goal = math.sqrt(pow(i[0]-goal[0],2)+pow(i[1]-goal[1],2))
        #     if (distance_to_goal <  math.sqrt(pow(new_position[0]-goal[0],2)+pow(new_position[1]-goal[1],2))):
        #         new_position = i
        new_position = self.helbling_modeling()
        self.model.grid.move_agent(self, new_position) ## 그 위치로 이동
        # self.kinetic_modeling()

    def kinetic_modeling(self):
        x = int(round(self.xy[0]))
        y = int(round(self.xy[1]))
        temp_loc = [(x-2, y), (x-1, y), (x+1, y), (x+2, y), (x, y+1), (x, y+2), (x, y-1), (x, y-2), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)] # (x,y)에 있는 agent와 상호작용하는 위치의 집합
        near_loc = []
        for i in temp_loc:
            if(i[0] > 0 and i[1] > 0 and i[0] < self.model.grid.width and i[1] < self.model.grid.height): # temp_loc에 있는 위치 중 map 벗어나는 위치 빼기
                near_loc.append(i) 
        near_agents_list = []
        for i in near_loc: #i -> (x,y) near_loc 안에 있는 agent들 확인
            near_agents = self.model.grid.get_cell_list_contents([i]) ## near_loc 에 존재하는 agents 모아
            if len(near_agents): 
                for near_agent in near_agents:
                    near_agents_list.append(near_agent) # 주변에 있는 agents을 모은다

        F_x = 0
        F_y = 0
        k = 1
        valid_distance = 3 # valid_distance안에 있는 agents와의 상호작용만 고려
        intend_force = 2.5 # 가고자 하는 힘
        time_step = 0.5 # 이 정도가 적절했다

        for near_agent in near_agents_list:
            n_x = near_agent.xy[0]
            n_y = near_agent.xy[1]
            d_x = self.xy[0] - n_x
            d_y = self.xy[1] - n_y
            d = math.sqrt(pow(d_x, 2) + pow(d_y, 2)) #각각 agent와의 거리
            if(valid_distance < d):
                continue    

            F = k * (valid_distance - d) #각 agent와의 반발력
            #print("F : ", F)
            if(d > 0 and near_agent.dead == False): #죽은 agent는 빼고 더해준다(각각 힘)
                F_x += (F*(d_x/d))
                F_y += (F*(d_y/d))
        #print(self.xy[0], self.xy[1])
        goal_x = central_of_goal(goal_list[check_stage(self.xy[0])])[0] - self.xy[0]
        goal_y = central_of_goal(goal_list[check_stage(self.xy[1])])[1] - self.xy[1]
        goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))
        if(goal_d != 0): ## goal 까지 거리가 남아있으면
            F_x += intend_force * (goal_x/goal_d)
            F_y += intend_force * (goal_y/goal_d) #intend force 분해

        self.acc[0] = F_x/self.mass 
        self.acc[1] = F_y/self.mass

        self.vel[0] = self.acc[0] ## s = vt + 0.5at^2, v = v+at 로 하니까 이상했대
        self.vel[1] = self.acc[1]

        self.xy[0] += self.vel[0] * time_step #객체의 다음 위치
        self.xy[1] += self.vel[1] * time_step
        
        next_x = int(round(self.xy[0])) #int 형변환
        next_y = int(round(self.xy[1]))

        if(next_x < 0): #아쉽다.. 고치길 바람 ## 밖에 있는 애들은 0에 있는 거라고 때려넣음..
            next_x = 0
        if(next_y < 0):
            next_y = 0
        #print(F_x, F_y)
        return (next_x, next_y)
    
    def helbling_modeling(self):
        from model_integrated import Model
        global random_disperse

        x = int(round(self.xy[0]))
        y = int(round(self.xy[1]))
        temp_loc = [(x-2, y), (x-1, y), (x+1, y), (x+2, y), (x, y+1), (x, y+2), (x, y-1), (x, y-2), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        near_loc = []
        for i in temp_loc:
            if(i[0]>0 and i[1]>0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
                near_loc.append(i)
        near_agents_list = []
        for i in near_loc:
            near_agents = self.model.grid.get_cell_list_contents([i])
            if len(near_agents):
                for near_agent in near_agents:
                    near_agents_list.append(near_agent) #kinetic 모델과 동일

        F_x = 0
        F_y = 0
        k = 1
        valid_distance = 3
        intend_force = 2.5
        time_step = 0.1# time step... 작게하면? 현실의 연속적인 시간과 비슷해져 현실적인 결과를 얻을 수 있음. 그러나 속도가 느려짐
                        # 크게하면? 속도가 빨라지나 비현실적.. (agent가 튕기는 등..)
        desired_speed = 2 # agent가 갈 수 있는 최대 속도, 나중에는 정규분포화 시킬 것
        repulsive_force = [0, 0]
        obstacle_force = [0, 0]
        for near_agent in near_agents_list:
            n_x = near_agent.xy[0]
            n_y = near_agent.xy[1]
            d_x = self.xy[0] - n_x
            d_y = self.xy[1] - n_y
            d = math.sqrt(pow(d_x, 2) + pow(d_y, 2))
            if(valid_distance<d):
                continue    

            F = k * (valid_distance-d)
            # print("F : ", F)
            # if(d>0 and near_agent.dead == False):
            #     F_x += (F*(d_x/d))
            #     F_y += (F*(d_y/d))
            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                repulsive_force[0] += k*np.exp(0.4/d)*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                repulsive_force[1] += k*np.exp(0.4/d)*(d_y/d)
            else :
                if(random_disperse):
                    repulsive_force = [50, -50]
                    random_disperse = 0
                else:
                    repulsive_force = [-50, 50] # agent가 정확히 같은 위치에 있을시 따로 떨어트리기 위함 
                    random_disperse = 1

        check_wall = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x, y-1), (x+1, y-1), (x+1, y), (x+1, y+1)]

        for i in check_wall: 
            print(len(self.model.wall_matrix))
            o_x = self.xy[0] - i[0]
            o_y = self.xy[1] - i[1]

            o_d = math.sqrt(pow(o_x, 2) + pow(o_y, 2))    
        
            if(i[0]>0 and i[1]>0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
                #print(len(self.model.wall_matrix))
                if(self.model.wall_matrix[i[0]][i[1]]): # agent 주위에 벽이 있으면..
                    obstacle_force[0] += k*np.exp(0.7/o_d)*(o_x/o_d) #벽으로 부터 힘을 받겠지
                    obstacle_force[1] += k*np.exp(0.7/o_d)*(o_y/o_d)
                        
                        

             
                
        #print(self.xy[0], self.xy[1])
        goal_x = central_of_goal(goal_list[check_stage(self.xy)])[0] - self.xy[0]
        goal_y = central_of_goal(goal_list[check_stage(self.xy)])[1] - self.xy[1]
        goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))

        if(goal_d != 0):
          desired_force = [intend_force*(desired_speed*(goal_x/goal_d)-self.vel[0]), intend_force*(desired_speed*(goal_y/goal_d)-self.vel[1])]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
          desired_force = [0, 0]
        
        
        #desried_force = intend_force(상수) * (가고자 했던 속도 - 현재 속도) 
        #가고자 했던 속도와 현재 속도가 차이가 많이 나면 #뛰어야겠지

        

        # if(goal_d != 0):
        #     F_x += intend_force * (goal_x/goal_d)
        #     F_y += intend_force * (goal_y/goal_d)

        F_x += desired_force[0]
        F_y += desired_force[1]

        F_x += obstacle_force[0]
        F_y += obstacle_force[1]

        self.acc[0] = F_x/self.mass
        self.acc[1] = F_y/self.mass

        self.vel[0] = self.acc[0]
        self.vel[1] = self.acc[1]

        self.xy[0] += self.vel[0] * time_step
        self.xy[1] += self.vel[1] * time_step
        
        next_x = int(round(self.xy[0]))
        next_y = int(round(self.xy[1]))

        if(next_x<0):
            next_x = 0
        if(next_y<0):
            next_y = 0
        if(next_x>199):
            next_x = 199
        if(next_y>199):
            next_y = 199
        #print(F_x, F_y)
        return (next_x, next_y)
        




