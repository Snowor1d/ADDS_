from mesa import Agent
import math
import numpy as np
import random
import copy

ATTACK_DAMAGE = 50
INITIAL_HEALTH = 100
HEALING_POTION = 20
exit_w = 5
exit_h = 5
exit_area = [[0,exit_w], [0, exit_h]]
STRATEGY = 1
random_disperse = 1

exit_area = [[0,exit_w], [0,exit_h]]

robot_xy = [2, 2]
robot_radius = 20 #로봇 반경 -> 10미터 
robot_status = 0
robot_ringing = 0


def space_connected_linear(xy1, xy2):
    check_connection = []
    for i1 in range(101):
        tmp = []
        for j1 in range(101):
            tmp.append(0)
        check_connection.append(tmp)
    
    for y in range(xy1[0][1]+1, xy1[1][1]):
        check_connection[xy1[0][0]][y] = 1 #left 
    for y in range(xy1[0][1]+1, xy1[1][1]):
        check_connection[xy1[1][0]][y] = 1 #right
    for x in range(xy1[0][0]+1, xy1[1][0]):
        check_connection[x][xy1[0][1]] = 1 #down
    for x in range(xy1[0][0]+1, xy1[1][0]):
        check_connection[x][xy1[1][1]] = 1 #up


    check_connection2 = copy.deepcopy(check_connection)
    checking = 0

    left_goal = [0, 0]
    left_goal_num = 0

    right_goal = [0, 0]
    right_goal_num = 0

    down_goal = [0, 0]
    down_goal_num = 0

    up_goal = [0, 0]
    up_goal_num = 0

    for y2 in range(xy2[0][1]+1, xy2[1][1]):
        check_connection2[xy2[0][0]][y2] += 1 #left 
        if(check_connection2[xy2[0][0]][y2] == 2): #space와 space2는 접한다 
            #print(space, space2, "가 LEFT에서 만남")
            left_goal[0] += xy2[0][0]
            left_goal[1] += y2
            left_goal_num = left_goal_num + 1
            checking = 1 #이을거다~ (space와 space2를 )
    for y3 in range(xy2[0][1]+1, xy2[1][1]):
        check_connection2[xy2[1][0]][y3] += 1 #right
        if(check_connection2[xy2[1][0]][y3] == 2):
            #print(space, space2, "가 RIGHT에서 만남")
            right_goal[0] += xy2[1][0]
            right_goal[1] += y3
            right_goal_num = right_goal_num + 1
            checking = 1
    for x2 in range(xy2[0][0]+1, xy2[1][0]):
        check_connection2[x2][xy2[0][1]] += 1 #down
        if(check_connection2[x2][xy2[0][1]] == 2):
            #print(space, space2, "가 DOWN에서 만남")
            down_goal[0] += x2
            down_goal[1] += xy2[0][1]
            down_goal_num = down_goal_num + 1
            checking = 1
    for x3 in range(xy2[0][0]+1, xy2[1][0]):
        check_connection2[x3][xy2[1][1]] += 1 #up
        if(check_connection2[x3][xy2[1][1]] == 2):
            #print(space, space2, "가 UP에서 만남")
            up_goal[0] += x3
            up_goal[1] += xy2[1][1]
            up_goal_num = up_goal_num + 1
            checking = 1

    if(left_goal[0] != 0 and left_goal[1] != 0):
        first_left_goal = [0, 0]
        first_left_goal[0] = (left_goal[0]/left_goal_num)
        first_left_goal[1] = (left_goal[1]/left_goal_num)
        return first_left_goal
        
    elif(right_goal[0] != 0 and right_goal[1] != 0):
        first_right_goal = [0, 0]
        first_right_goal[0] = (right_goal[0]/right_goal_num)
        first_right_goal[1] = (right_goal[1]/right_goal_num)
        return first_right_goal
    
    elif(down_goal[0] != 0 and down_goal[1] != 0):
        first_down_goal = [0, 0]
        first_down_goal[0] = (down_goal[0]/down_goal_num)
        first_down_goal[1] = (down_goal[1]/down_goal_num)
        return first_down_goal
 

    elif(up_goal[0] != 0 and up_goal[1] != 0):
        first_up_goal = [0, 0]
        first_up_goal[0] = (up_goal[0]/up_goal_num)
        first_up_goal[1] = (up_goal[1]/up_goal_num)
        return first_up_goal




# goal_list = [[(80, 119), (79, 119), (78, 119), (77, 119)], #gate1
#              [(90, 120)], #gate2
#              [(198, 119), (197, 119), (196, 119)], #gate3
#              [(119, 90)], #gate4
#              [(119, 19), (119, 20), (119, 21), (119, 22)], #gate5 
#              [(0,0), (0,1), (1,0), (0,0)] ] #gate6 

# goal_list = [[(79, 119), (78, 119)],
#                [(90, 120)], 
#              [(198, 119), (197, 119)],
#              [(119, 90)],
#              [(119, 19), (119, 20)],
#              [(0,0), (0,1), (1,0), (0,0)] 
#              ]

# goal_list = [[(198, 60), (199, 60), (197, 60), (196, 60), (195, 60), (194, 60)
#               ,(198, 59), (199, 59), (197, 59)], [(0,0), (0,1), (1,0), (1,1)]]
#goal_list = [[(0,0), (0,1)]]
goal_list = [[(71, 52)], [(89, 52)]]


def check_stage(pose):
    # stage_1 = [[60,200], [60,200]] #    60 < x범위 < 200
    #                                #    60 < y범위 < 200
    #                                # stage_2 는, else
    # if(pose[0]>stage_1[0][0] and pose[0]<stage_1[0][1] and pose[1]>stage_1[1][0] and pose[1]<stage_1[1][1]):
    #     return 0
    # else:
    #     return 1

    stage_1 = [[40, 68], [20, 80]]
    stage_2 = [[69, 120], [120, 200]]
    stage_3 = [[120, 200], [120, 200]]
    stage_4 = [[120, 200], [80, 120]]
    stage_5 = [[120, 200], [0, 80]]

    x = pose[0]
    y = pose[1]

    if(pose[0]>stage_1[0][0] and pose[0]<stage_1[0][1] and pose[1]>stage_1[1][0] and pose[1]<stage_1[1][1]):
        return 0
    else:
        return 1

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

    def __init__(self, unique_id, model, pos, type): 
        super().__init__(unique_id, model)
        global robot_xy
        robot_xy = pos
        self.goal_init = 0
        self.type = type
        self.health = INITIAL_HEALTH
        self.attack_damage = ATTACK_DAMAGE
        self.attacked = False
        self.dead = False
        self.robot_guide = 0
        self.drag = 0
        self.dead_count = 0
        self.buried = False
        self.which_goal = 0
        self.previous_stage = []
        self.now_goal = [0,0]
        
        #self.robot_xy = [2,2]
        #self.robot_status = 0

        self.xy = pos
        self.vel = [0, 0]
        self.acc = [0, 0]
        self.mass = 3
        self.previous_goal = [0,0]

        #for robot 
        self.robot_space = ((0,0), (5,95))
        self.mission_complete = 1
        self.going = 0
        self.guide = 0
        self.save_target = 0
        self.save_point = 0
        self.robot_now_path = []
        self.robot_waypoint_index = 0

        self.go_path_num= 0
        self.back_path_num = 0


        # self.xy[0] = self.random.randrange(self.model.grid.width)
        # self.xy[1] = self.random.randrange(self.model.grid.height)
        
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
        if(self.type != 3): #robot은 죽지 않는다
            if (self.model.is_left_exit):
                if (self.xy[0] > self.model.left_exit_area[0][0] and self.xy[0] < self.model.left_exit_area[1][0] and self.xy[1] > self.model.left_exit_area[0][1] and self.xy[1]<self.model.left_exit_area[1][1]):
                    self.health = 0
                    self.dead = True 

            if (self.model.is_right_exit):
                if (self.xy[0] > self.model.right_exit_area[0][0] and self.xy[0] < self.model.right_exit_area[1][0] and self.xy[1] > self.model.right_exit_area[0][1] and self.xy[1]<self.model.right_exit_area[1][1]):
                    self.health = 0
                    self.dead = True 
        
            if (self.model.is_up_exit):
                if (self.xy[0] > self.model.up_exit_area[0][0] and self.xy[0] < self.model.up_exit_area[1][0] and self.xy[1] > self.model.up_exit_area[0][1] and self.xy[1]<self.model.up_exit_area[1][1]):
                    self.health = 0
                    self.dead = True 
        
            if (self.model.is_down_exit):
                if (self.xy[0] > self.model.down_exit_area[0][0] and self.xy[0] < self.model.down_exit_area[1][0] and self.xy[1] > self.model.down_exit_area[0][1] and self.xy[1]<self.model.down_exit_area[1][1]):
                    self.health = 0
                    self.dead = True 
        # if (check_departure([self.xy[0],self.xy[1]], goal_list[len(goal_list)-1])):
        #     self.dead = True
        #     self.health = 0 ## 이게 0이어야 current healthy agent 수에 포함이 안 됨 ~!

        # if (self.which_goal != (len(goal_list)-1)):
        #     if(check_departure([self.xy[0], self.xy[1]], goal_list[self.which_goal])):
        #         self.which_goal += 1

        self.move()

    def check_stage_agent(self):
        x = self.xy[0]
        y = self.xy[1]
        now_stage = []
        for i in self.model.space_list:
            if (x>i[0][0] and x<i[1][0] and y>i[0][1] and y<i[1][1]):
                now_stage = i
                break
        if(len(now_stage) != 0):
            now_stage = ((now_stage[0][0], now_stage[0][1]), (now_stage[1][0], now_stage[1][1]))
        else:
            now_stage = ((0,0), (5, 95))
        return now_stage

    def which_goal_agent_want(self):
        

        if(self.goal_init == 0):
            now_stage = self.check_stage_agent()
            goal_candiate = self.model.space_goal_dict[now_stage]
            if(len(goal_candiate)==1):
                goal_index = 0
            else:
                goal_index = random.randint(0, len(goal_candiate)-1)
            self.now_goal = goal_candiate[goal_index]
            self.goal_init = 1
            self.previous_stage = now_stage
        now_stage = self.check_stage_agent() #now_stage -> agent가 현재 어느 stage에
        if(self.previous_stage != self.check_stage_agent()):
            goal_candiate = self.model.space_goal_dict[now_stage] # ex) [[2,0], [3,5],[4,1]] 
            goal_candiate2 = []
            if(len(goal_candiate)>1):
                min_d = 1000
                min_i = goal_candiate[0]
                for i in goal_candiate:
                    d = math.sqrt(pow(self.xy[0]-i[0], 2)+pow(self.xy[1]-i[1], 2))
                    if (d<min_d):
                        min_d = d
                        min_i = i #가장 가까운 골 찾기
                for j in goal_candiate: 
                    if (j==min_i): #goal 후보에서 빼버림
                        continue
                    else :
                        goal_candiate2.append(j)
                if(len(goal_candiate2)==1):
                    goal_index = 0
                else:
                    goal_index = random.randint(0, len(goal_candiate2)-1)
                self.now_goal = goal_candiate2[goal_index]
                self.previous_stage = now_stage
                return
            elif(len(goal_candiate)==0):
                self.now_goal = self.previous_goal
            else :
                goal_index = 0
                self.now_goal = goal_candiate[goal_index]
                self.previous_stage = now_stage
            self.previous_goal = self.now_goal

            


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

        # possible_steps = self.model.grid.get_neighborhood( ## 다음 step에서 갈 수 있는 곳은 이웃 grid
        #     self.pos, moore=True, include_center=False ##? 
        # )

        cells_with_agents = []
        # looking for agents in the cells around the agent
        # for cell in possible_steps:
        #     otherAgents = self.model.grid.get_cell_list_contents([cell])
        #     if len(otherAgents): ## 주변에 agent 있니?
        #         for agent in otherAgents: ## 안 죽은 agent 들 cells_with_agents에 추가
        #             if not agent.dead:
        #                 cells_with_agents.append(agent)

        # if there is some agent on the neighborhood
        # if len(cells_with_agents): ## 주변 agent 수 만큼
        #     if STRATEGY == 1: ## 언제 1 되냐???
        #         self.attackOrMove(cells_with_agents, possible_steps)
        #     else: ## 주변에 있는 애들 attack
        #         self.attack(cells_with_agents)
        # new_position = possible_steps[0]
        # for i in possible_steps:
        #     distance_to_goal = math.sqrt(pow(i[0]-goal[0],2)+pow(i[1]-goal[1],2))
        #     if (distance_to_goal <  math.sqrt(pow(new_position[0]-goal[0],2)+pow(new_position[1]-goal[1],2))):
        #         new_position = i
        if (self.type == 3):
            new_position = self.robot_policy()
            self.model.grid.move_agent(self, new_position)
            return

        new_position = self.test_modeling()
        if(self.type ==0 or self.type==1):
            self.model.grid.move_agent(self, new_position) ## 그 위치로 이동
        # self.kinetic_modeling()

    def kinetic_modeling(self):
        x = int(round(self.xy[0]))
        y = int(round(self.xy[1]))
        #temp_loc = [(x-2, y), (x-1, y), (x+1, y), (x+2, y), (x, y+1), (x, y+2), (x, y-1), (x, y-2), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)] # (x,y)에 있는 agent와 상호작용하는 위치의 집합
        temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
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
        from model_renew import Model
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
        return (next_x, next_y)
    
    def robot_policy(self):
        time_step = 0.2
        from model_renew import Model
        global random_disperse
        global robot_status
        global robot_xy 
        global robot_radius
        global robot_ringing
        self.drag = 1
        robot_status = 1
        space_agent_num = self.agents_in_each_space() #어느 stage에 몇명이 있는지
        floyd_distance = self.model.floyd_distance # floyd_distance[stage1][stage2] = 최단거리 
        floyd_path = self.model.floyd_path #floyd_path[stage1][stage2] = stage_x 
                                           #floyd_path[stage_x][stage_2] = stage_y 
                                           #floyd_path[stage_y][stage_2] = ste... 
                                           # s1 -> sx -> sy -> .. stage
    
        

        self.robot_space = self.model.grid_to_space[int(robot_xy[0])][int(robot_xy[1])] #로봇이 어느 stage에 있는지 나온다 

        if(self.mission_complete == 1): #새로운 탈출 path를 찾는다
            self.robot_now_path = [] # [[1,3], [4,5], [5,1]] 
            agent_max = 0 #agent가 가장 많은 stage 
            for i in space_agent_num.keys(): 
                if (space_agent_num[i]>agent_max):
                    self.save_target = i #현재 가장 인구가 많이 있는 stage
                    agent_max = space_agent_num[self.save_target] 
            
            evacuation_points = []
            if(self.model.is_left_exit): 
                evacuation_points.append(((0,0), (5, 95)))
            if(self.model.is_up_exit):
                evacuation_points.append(((0,95), (95, 99)))
            if(self.model.is_right_exit):
                evacuation_points.append(((95,5), (99, 99)))
            if(self.model.is_down_exit):
                evacuation_points.append(((5,0), (99, 5))) #evacuation_points에 탈출구들 저장 

            min_distance = 1000
            for i in evacuation_points: #space_target에서 가장 가까운 탈출구를 찾기 
                if(floyd_distance[self.save_target][i]<min_distance):
                    self.save_point = i 
            go_path = self.model.get_path(floyd_path, self.robot_space, self.save_target) #로봇의 초기 위치 -> save_target까지 가는데 최단 경로 stage 리스트 
            
            back_path = self.model.get_path(floyd_path, self.save_target, self.save_point) # save_target(인구가 가장 많은 곳)에서 save_point(safe zone) 까지의 최단 경로 
            self.go_path_num = len(go_path) #guide를 하기 위해서 ~ 

            for i in range(len(go_path)-1):
                self.robot_now_path.append(space_connected_linear(go_path[i], go_path[i+1])) #(stage1 stage2) -> 중간 goal을 알려준다  
            self.robot_now_path.append([(self.save_target[0][0]+self.save_target[1][0])/2, (self.save_target[0][1]+self.save_target[1][1])/2]) #save target 중점까지 간다 
            for i in range(len(back_path)-1):
                self.robot_now_path.append(space_connected_linear(back_path[i], back_path[i+1]))  #back path도 넣는다 
            self.mission_complete = 0 
        #print(self.robot_now_path)
        
        if(self.robot_waypoint_index > self.go_path_num-1): # 돌아오는 상황 
            robot_status = 1 #robot_status가 1일때 -> guide함, 로봇 색깔바뀜(빨간색), 로봇에 영향받는 agent 색깔 바뀜(주황색) 
            self.drag = 1 
        else:
            robot_status = 0
            self.drag = 0
        print("현재 골 : ", self.robot_now_path[self.robot_waypoint_index])
        d = (pow(self.robot_now_path[self.robot_waypoint_index][0]-robot_xy[0],2) + pow(self.robot_now_path[self.robot_waypoint_index][1]-robot_xy[1],2)) #현재 위치와 goal까지의 거리 구하기
        if (d<1):
            self.robot_waypoint_index = self.robot_waypoint_index + 1

        if(self.robot_waypoint_index == len(self.robot_now_path)):
            self.mission_complete = 1 #미션을 새로 만들어야해 (끝났으니까)
            self.robot_waypoint_index = 0
            return [int(robot_xy[0]), int(robot_xy[1])]

        goal_x = self.robot_now_path[self.robot_waypoint_index][0] - robot_xy[0] #역학을 위한.. 
        goal_y = self.robot_now_path[self.robot_waypoint_index][1] - robot_xy[1]
        goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))
        
        intend_force = 2
        desired_speed = 1.5

        if(self.drag == 0):
            desired_speed = 5
        else:
            desired_speed = 5

        if(goal_d != 0):
            desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
            desired_force = [0, 0]
    
        
        x=int(round(robot_xy[0]))
        y=int(round(robot_xy[1]))

        #temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
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
        repulsive_force = [0, 0]
        obstacle_force = [0, 0]

        k=4

        for near_agent in near_agents_list:
            n_x = near_agent.xy[0]
            n_y = near_agent.xy[1]
            d_x = robot_xy[0] - n_x
            d_y = robot_xy[1] - n_y
            d = math.sqrt(pow(d_x, 2) + pow(d_y, 2))


            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0

                elif(near_agent.type == 1): ## agents
                    repulsive_force[0] += 0/4*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 0/4*np.exp(-(d/2))*(d_y/d) 

                elif(near_agent.type == 11):## 검정벽 
                    repulsive_force[0] += 2*k*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 2*k*np.exp(-(d/2))*(d_y/d)
            else :
                if(random_disperse):
                    repulsive_force = [1, -1]
                    random_disperse = 0
                else:
                    repulsive_force = [-1, 1] # agent가 정확히 같은 위치에 있을시 따로 떨어트리기 위함 
                    random_disperse = 1
        
        F_x = 0
        F_y = 0
        
        F_x += desired_force[0]
        F_y += desired_force[1]

        F_x += repulsive_force[0]
        F_y += repulsive_force[1]
        vel = [0,0]
        vel[0] = F_x/self.mass
        vel[1] = F_y/self.mass

        robot_xy[0] += vel[0] * time_step
        robot_xy[1] += vel[1] * time_step
        
        next_x = int(round(robot_xy[0]))
        next_y = int(round(robot_xy[1]))

        if(next_x<0):
            next_x = 0
        if(next_y<0):
            next_y = 0
        if(next_x>99):
            next_x = 99
        if(next_y>99):
            next_y = 99
        #print(F_x, F_y)
            
        #if(self.dead != True):
            #print(self.now_goal)
            #print(desired_force[0], desired_force[1])
            #print(F_x, F_y)

        #self.robot_guide = 0
        return (next_x, next_y)

    def agents_in_each_space(self):
        from model_renew import Model
        space_agent_num = {}
        for i in self.model.space_list:
            space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0
        for i in self.model.agents:
            space_xy = self.model.grid_to_space[int((i.xy)[0])][int((i.xy)[1])]
            if(i.dead == False):
                space_agent_num[((space_xy[0][0], space_xy[0][1]), (space_xy[1][0], space_xy[1][1]))] +=1 
        #for j in space_agent_num.keys():
            #print(j, "공간에 ", space_agent_num[j], "명이 있음")
        return space_agent_num


        
    def test_modeling(self):
        global robot_radius
        global robot_xy
        global robot_status
        from model_renew import Model
        global random_disperse

        x = int(round(self.xy[0]))
        y = int(round(self.xy[1]))
        #temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
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
        k = 3
        r_0 = 0.3
        valid_distance = 3
        intend_force = 2
        time_step = 0.2 #time step... 작게하면? 현실의 연속적인 시간과 비슷해져 현실적인 결과를 얻을 수 있음. 그러나 속도가 느려짐
                        # 크게하면? 속도가 빨라지나 비현실적.. (agent가 튕기는 등..)
        #time_step마다 desired_speed로 가고, desired speed의 단위는 1픽셀, 1픽셀은 0.5m
        #만약 time_step가 0.1이고, desired_speed가 2면.. 0.1초 x 2x0.5m = 한번에 최대 0.1m 이동 가능..
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
                
            # if(d!=0):
            #     if(near_agent.type != 11):
            #         repulsive_force[0] += k*np.exp(-pow((r_0/d), 2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
            #         repulsive_force[1] += k*np.exp(-pow((r_0/d), 2))*(d_y/d)
            #     else:
            #         repulsive_force[0] += 10*k*np.exp(-pow((r_0/d), 2))*(d_x/d)
            #         repulsive_force[1] += 10*k*np.exp(-pow((r_0/d), 2))*(d_y/d)
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0

                elif(near_agent.type == 1): ## agents
                    repulsive_force[0] += 1/4*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 1/4*np.exp(-(d/2))*(d_y/d) 

                elif(near_agent.type == 11):## 검정벽 
                    repulsive_force[0] += 2*k*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 2*k*np.exp(-(d/2))*(d_y/d)
            else :
                if(random_disperse):
                    repulsive_force = [1, -1]
                    random_disperse = 0
                else:
                    repulsive_force = [-1, 1] # agent가 정확히 같은 위치에 있을시 따로 떨어트리기 위함 
                    random_disperse = 1

        # check_wall = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x, y-1), (x+1, y-1), (x+1, y), (x+1, y+1)]

        # for i in check_wall: 
        #     o_x = self.xy[0] - i[0]
        #     o_y = self.xy[1] - i[1]

        #     o_d = math.sqrt(pow(o_x, 2) + pow(o_y, 2))    
        
        #     if(i[0]>0 and i[1]>0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
        #         #print(len(self.model.wall_matrix))
        #         if(self.model.wall_matrix[i[0]][i[1]]): # agent 주위에 벽이 있으면..
        #             obstacle_force[0] += k*np.exp(0.7/o_d)*(o_x/o_d) #벽으로 부터 힘을 받겠지
        #             obstacle_force[1] += k*np.exp(0.7/o_d)*(o_y/o_d)
                        
                        

             
        self.which_goal_agent_want()
        # goal_x = central_of_goal(goal_list[check_stage(self.xy)])[0] - self.xy[0]
        # goal_y = central_of_goal(goal_list[check_stage(self.xy)])[1] - self.xy[1]
        # goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))
        goal_x = self.now_goal[0] - self.xy[0]
        goal_y = self.now_goal[1] - self.xy[1]
        goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))

        robot_x = robot_xy[0] - self.xy[0]
        robot_y = robot_xy[1] - self.xy[1]
        robot_d = math.sqrt(pow(robot_x,2)+pow(robot_y,2))
        if(robot_d<robot_radius and robot_status == 1):
            goal_x = robot_x
            goal_y = robot_y
            goal_d = robot_d
            self.type = 1
        else :
            self.type = 0


        # if(self.unique_id == 0):
        #     goal_x = goal_list[0][0][0] - self.xy[0]
        #     goal_y = goal_list[0][0][1] - self.xy[1]
        #     goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))
        # else:
        #     goal_x = goal_list[1][0][0] - self.xy[0]
        #     goal_y = goal_list[1][0][1] - self.xy[1]
        #     goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))

        if(goal_d != 0):
          desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
          desired_force = [0, 0]
        
        
        #desried_force = intend_force(상수) * (가고자 했던 속도 - 현재 속도) 
        #가고자 했던 속도와 현재 속도가 차이가 많이 나면 #뛰어야겠지

        

        # if(goal_d != 0):
        #     F_x += intend_force * (goal_x/goal_d)
        #     F_y += intend_force * (goal_y/goal_d)

        F_x += desired_force[0]
        F_y += desired_force[1]

        #F_x += obstacle_force[0]
        #F_y += obstacle_force[1]
        
        F_x += repulsive_force[0]
        F_y += repulsive_force[1]
        

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
        if(next_x>99):
            next_x = 99
        if(next_y>99):
            next_y = 99
        #print(F_x, F_y)
            
        #if(self.dead != True):
            #print(self.now_goal)
            #print(desired_force[0], desired_force[1])
            #print(F_x, F_y)

        self.robot_guide = 0
        return (next_x, next_y)




