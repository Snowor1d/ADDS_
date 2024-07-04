#this source code requires Mesa==2.2.1 
#^__^
from mesa import Agent
import math
import numpy as np
import random
import copy
num_remained_agent = 0
NUMBER_OF_CELLS = 50 


one_foot = 1
SumList = [0, 0, 0, 0, 0]
DifficultyList = [0, 0, 0, 0, 0]

ATTACK_DAMAGE = 50
INITIAL_HEALTH = 100
HEALING_POTION = 20
exit_w = 5
exit_h = 5
exit_area = [[0,exit_w], [0, exit_h]]
STRATEGY = 1
random_disperse = 1

theta_1 = random.randint(1,10)
theta_2 = random.randint(1,10)
theta_3 = random.randint(1,10)

exit_area = [[0,exit_w], [0,exit_h]]

robot_xy = [2, 2]
robot_radius = 5 #로봇 반경 -> 10미터 
robot_status = 0
robot_ringing = 0
robot_goal = [0, 0]
past_target = ((0,0), (0,0))


def Multiple_linear_regresssion(distance_ratio, remained_ratio, now_affected_agents_ratio, v_min, v_max):
    global theta_1, theta_2, theta_3
    v = distance_ratio*theta_1 + remained_ratio*theta_2 + now_affected_agents_ratio*theta_3
    if (v>v_max):
        return v_max
    elif (v<v_min):
        return v_min
    else:
        return v




goal_list = [[(71, 52)], [(89, 52)]]

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



class FightingAgent2(Agent):
    """An agent that fights."""

    def __init__(self, unique_id, model, pos, type): 
        #print("pos : ", pos)
        super().__init__(unique_id, model)
        global robot_xy
        self.robot_step = 0
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

        self.now_action = ["UP", "GUIDE"]

        #for robot 
        self.robot_space = ((0,0), (5,45))
        self.mission_complete = 1
        self.going = 0
        self.guide = 0
        self.save_target = 0
        self.save_point = 0
        self.robot_now_path = []
        self.robot_waypoint_index = 0

        self.go_path_num= 0
        self.back_path_num = 0

        file_path = 'weight.txt'
        file = open(file_path, "r")
        
        lines = file.readlines()

        file.close()

        self.w1 = float(lines[0].strip())
        self.w2 = float(lines[1].strip())
        self.w3 = float(lines[2].strip())
        self.w4 = float(lines[3].strip())
        self.w5 = float(lines[2].strip())
        self.w6 = float(lines[3].strip()) 

        self.feature_weights_guide = [self.w1, self.w2, self.w3]
        self.feature_weights_not_guide = [self.w4, self.w5, self.w6]


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
        self.move()

    def check_stage_agent(self): ## 이건 언제 쓰이나??? agent 움직일 때 현재 자기가 있는 위치 알 때
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
            now_stage = ((0,0), (5, 45))
        return now_stage

    def which_goal_agent_want(self):
        now_stage = self.check_stage_agent()

        if(self.goal_init == 0):
            goal_candiate = self.model.space_goal_dict[now_stage]
            if(len(goal_candiate)==1):
                goal_index = 0
            else:
                goal_index = random.randint(0, len(goal_candiate)-1)
            self.now_goal = goal_candiate[goal_index]
            self.goal_init = 1
            self.previous_stage = now_stage
        if(self.previous_stage != now_stage):
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
                    else:
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
        agentToAttack.attacked = True ## 맞은 애 attacked 됐다~ 
        if agentToAttack.health <= 0: ## health 가 0보다 작으면 dead
            agentToAttack.dead = True

    def move(self) -> None:
        global goal_list
        """Handles the movement behavior.
        Here the agent decides   if it moves,
        drinks the heal potion,
        or attacks other agent."""

        cells_with_agents = []

        if (self.type == 3):
            self.robot_step += 1
            #print(self.model.dict_NoC)
            robot_space_tuple = tuple(map(tuple, self.robot_space))
            if(self.model.dict_NoC[robot_space_tuple]==1):
                new_position = self.model.robot_respawn()
            new_position = self.robot_policy_Q()

            self.model.reward_distance_difficulty()
            
            
            #new_position = self.model.robot_respawn()
            reward = self.reward_distance(robot_xy, "none", "none")
            #print("reward : ", reward)
            self.reward_difficulty_space(robot_xy, "none", "none") ###여기야!!!!!!! 여기다가 reward 계산해야댕
            #reward += self.reward_difficulty_space

## 여기 아래에서 weight 변경하는 코드 넣어주세용
            

            #print("weights update ~~ ^^ ", self.w1, self.w2, self.w3, self.w4, self.w5, self.w6)
            ""
            #self.update_weight(reward)

            # self.model.step() 

            # from ADDS_AS import n
            if self.robot_step == 500 or num_remained_agent == 0:
            # if num_remained_agent == 0:
                file2 = open("weight.txt", 'w')
                new_lines = [str(self.w1) + '\n', str(self.w2) + '\n', str(self.w3) + '\n', str(self.w4) + '\n', str(self.w5) + '\n', str(self.w6) + '\n']
                file2.writelines(new_lines)
                file2.close()


            self.model.grid.move_agent(self, new_position)
            return

        new_position = self.test_modeling()
        if(self.type ==0 or self.type==1):
            self.model.grid.move_agent(self, new_position) ## 그 위치로 이동
  
    
    def agents_in_each_space(self):
        global num_remained_agent
        #from model import Model
        space_agent_num = {}
        for i in self.model.space_list:
            space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0
        for i in self.model.agents:
            space_xy = self.model.grid_to_space[int(round((i.xy)[0]))][int(round((i.xy)[1]))]
            if(i.dead == False and (i.type==0 or i.type==1)):
                space_agent_num[((space_xy[0][0], space_xy[0][1]), (space_xy[1][0], space_xy[1][1]))] +=1 
        
        for j in space_agent_num.keys():
            num_remained_agent += space_agent_num[j]
        return space_agent_num


    def agents_in_each_space2(self): #this function will not change global variable num_remained_agent 
        global num_remained_agent
        #from model import Model
        space_agent_num = {}
        for i in self.model.space_list:
            space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0
        for i in self.model.agents:
            space_xy = self.model.grid_to_space[int(round((i.xy)[0]))][int(round((i.xy)[1]))]
            if(i.dead == False and (i.type==0 or i.type==1)):
                space_agent_num[((space_xy[0][0], space_xy[0][1]), (space_xy[1][0], space_xy[1][1]))] +=1 
        return space_agent_num
    

    def agents_in_robot_area(self, robot_xyP):
        #from model import Model
        number_a = 0
        for i in self.model.agents:
            if(i.dead == False and (i.type == 0 or i.type == 1)): ##  agent가 살아있을 때 / 끌려가는 agent 일 때
                if pow(robot_xyP[0]-i.xy[0], 2) + pow(robot_xyP[1]-i.xy[1], 2) < pow(robot_radius, 2) : ## 로봇 반경 내에 agent가 있다면
                    number_a += 1

        return number_a

        

    def find_target(self, space_agent_num, floyd_distance):
        global past_target
        self.robot_now_path = []
        agent_max = 0
        space_priority = {}
        distance_to_safe = {}
        
        evacuation_points = [self.model.exit_compartment] #evacuation_points에 탈출구들 저장 

        # evacuation_points = []
        # if(self.model.is_left_exit): 
        #     evacuation_points.append(((0,0), (5, 45)))
        # if(self.model.is_up_exit):
        #     evacuation_points.append(((0,45), (45, 49)))
        # if(self.model.is_right_exit):
        #     evacuation_points.append(((45,5), (49, 49)))
        # if(self.model.is_down_exit):
        #     evacuation_points.append(((5,0), (49, 5))) #evacuation_points에 탈출구들 저장 

        for i in space_agent_num.keys():
            min_d = 10000
            distance_to_safe[i] = min_d
            for j in evacuation_points:
                if min_d>floyd_distance[i][j] :
                    min_d = floyd_distance[i][j]
                    distance_to_safe[i] = min_d
        for i2 in space_agent_num.keys():
            if (distance_to_safe[i2]>9999):
                distance_to_safe[i2] = -1

                
        
        for l in space_agent_num.keys():
            space_priority[l] = distance_to_safe[l] * space_agent_num[l]
            if(l==past_target):
                space_priority[l] -= 10000
        #(space_priority)
        agent_max = 0
        for k in space_priority.keys():
            if (space_priority[k]>agent_max):
                self.save_target = k
                agent_max = space_priority[self.save_target]
        min_distance = 1000
        for m in evacuation_points: #space_target에서 가장 가까운 탈출구를 찾기 
            if(floyd_distance[self.save_target][m]<min_distance):
                self.save_point = m
                min_distance = floyd_distance[self.save_target]

        
    def test_modeling(self):
        global robot_radius
        global robot_xy
        global robot_status
        #from model import Model
        global random_disperse

        x = int(round(self.xy[0]))
        y = int(round(self.xy[1]))
        temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        #temp_loc = [(x-2, y), (x-1, y), (x+1, y), (x+2, y), (x, y+1), (x, y+2), (x, y-1), (x, y-2), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
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
            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0

                elif(near_agent.type == 1): ## agents
                    repulsive_force[0] += 1*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 1*np.exp(-(d/2))*(d_y/d) 

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

             
        self.which_goal_agent_want() # agent의 다음 골 목표 설정 

        goal_x = self.now_goal[0] - self.xy[0]
        goal_y = self.now_goal[1] - self.xy[1]
        goal_d = math.sqrt(pow(goal_x,2)+pow(goal_y,2))



        if(goal_d != 0):
          desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
          desired_force = [0, 0]
        

        F_x += desired_force[0]
        F_y += desired_force[1]
        
        F_x += repulsive_force[0]
        F_y += repulsive_force[1]
        

        self.acc[0] = F_x/self.mass
        self.acc[1] = F_y/self.mass

        self.vel[0] = self.acc[0]
        self.vel[1] = self.acc[1]

        self.xy[0] += self.vel[0] * time_step
        self.xy[1] += self.vel[1] * time_step
        #self.xy = self.move_to_valid(self.xy)
        
        next_x = int(round(self.xy[0]))
        next_y = int(round(self.xy[1]))

        if(next_x<0):
            next_x = 0
        if(next_y<0):
            next_y = 0
        if(next_x>49):
            next_x = 49
        if(next_y>49):
            next_y = 49


        self.robot_guide = 0
        return (next_x, next_y)
    
    def move_to_valid(self, loc):
        original_loc = [0, 0]
        original_loc[0] = loc[0]
        original_loc[1] = loc[1]
        count = 0
        #print("함수 내 : ", self.model.valid_space[int(round(loc[0]))][int(round(loc[1]))])
        while(self.model.valid_space[int(round(loc[0]))][int(round(loc[1]))]==0):
            loc[0] = (original_loc[0] - 0.5)
            loc[1] = (original_loc[1] - 0.5)
            loc[0] += (random.randint(0, 5)/5)
            loc[1] += (random.randint(0, 5)/5)
            count += 1
            if (count>=40):
                break 
        if (count>=40):
            while(self.model.valid_space[int(round(loc[0]))][int(round(loc[1]))]==0):
                #print("두번째 루프")
                loc[0] = (original_loc[0] - 1)
                loc[1] = (original_loc[1] - 1)
                loc[0] += (random.randint(0, 10)/5)
                loc[1] += (random.randint(0, 10)/5)
                count += 1 
                if (count>=100):
                    break 
        if (count>=100):
            while(self.model.valid_space[int(round(loc[0]))][int(round(loc[1]))]==0):
                loc[0] = (original_loc - 2)
                loc[1] = (original_loc - 2)
                loc[0] += (random.randint(0, 20)/5)
                loc[1] += (random.randint(0, 20)/5)
                count += 1
                if (count>=200):
                    break 
        return loc 

    
    def move_to_valid_robot(self):
        global robot_xy
        original_loc = [0, 0]
        original_loc[0] = robot_xy[0]
        original_loc[1] = robot_xy[1]
        count = 0
        #print("함수 내 : ", self.model.valid_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))])
        while(self.model.valid_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]==0):
            robot_xy[0] = (original_loc[0] - 0.5)
            robot_xy[1] = (original_loc[1] - 0.5)
            robot_xy[0] += (random.randint(0, 5)/5)
            robot_xy[1] += (random.randint(0, 5)/5)
            count += 1
            if (count>=40):
                break 
        if (count>=40):
            while(self.model.valid_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]==0):
                robot_xy[0] = (original_loc[0] - 1)
                robot_xy[1] = (original_loc[1] - 1)
                robot_xy[0] += (random.randint(0, 10)/5)
                robot_xy[1] += (random.randint(0, 10)/5)
                count += 1 
                if (count>=100):
                    break 
        if (count>=100):
            while(self.model.valid_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]==0):
                robot_xy[0] = (original_loc - 2)
                robot_xy[1] = (original_loc - 2)
                robot_xy[0] += (random.randint(0, 20)/5)
                robot_xy[1] += (random.randint(0, 20)/5)
                count += 1
                if (count>=200):
                    break 

        



    def F1_distance(self, state, action, mode):
        from model import space_connected_linear
        global robot_xy
        global one_foot

        now_space = self.model.grid_to_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]
        evacuation_points = []
        if(self.model.is_left_exit): 
            evacuation_points.append(((0,0), (5, 45)))
            exit = ((0,0), (5, 45))
        if(self.model.is_up_exit):
            evacuation_points.append(((0,45), (45, 49)))
            exit = ((0,45), (45, 49))
        if(self.model.is_right_exit):
            evacuation_points.append(((45,5), (49, 49)))
            exit = ((45,5), (49, 49))
        if(self.model.is_down_exit):
            evacuation_points.append(((5,0), (49, 5))) #evacuation_points에 탈출구들 저장 ##?? 근데 왜 초록 구역이 아니고 사이드 구역 전체일까? 그것은 여기로 오면 탈출하기 때문이지요?
            exit = ((5,0), (49, 5))
        min_distance = 1000
        floyd_distance = self.model.floyd_distance 

        
        next_vertex_matrix = self.model.floyd_warshall()[0] ## 이중 딕셔너리, start space 부터 end space 까지의 경로
        for i in evacuation_points: #space_target에서 가장 가까운 탈출구를 찾기 
            if(floyd_distance[((now_space[0][0],now_space[0][1]), (now_space[1][0], now_space[1][1]))][i] < min_distance): ##floyd_distance[start(now space)][end(i;출구)] 
                exit = i ##지금은 출구가 하나라서 exit이 evacuation_points이지만 출구가 많아진다면 ~
        
        if(exit!=0):
            next_goal = space_connected_linear(((now_space[0][0],now_space[0][1]), (now_space[1][0], now_space[1][1])), next_vertex_matrix[((now_space[0][0],now_space[0][1]), (now_space[1][0], now_space[1][1]))][exit])
        else :
            next_goal = robot_xy
        now_space_x_center = (now_space[0][0] + now_space[1][0])/2
        now_space_y_center = (now_space[1][0] + now_space[1][1])/2

        next_robot_position = [0, 0]
        next_robot_position[0] += robot_xy[0]
        next_robot_position[1] += robot_xy[1]

        if (action=="UP"):
            next_robot_position[1] += one_foot
        elif (action=="DOWN"):
            next_robot_position[1] -= one_foot
        elif (action=="LEFT"):
            next_robot_position[0] -= one_foot
        elif (action=="RIGHT"):
            next_robot_position[0] += one_foot

        result = floyd_distance[((now_space[0][0],now_space[0][1]), (now_space[1][0], now_space[1][1]))][exit] - math.sqrt(pow(now_space_x_center-next_goal[0],2)+pow(now_space_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-next_robot_position[0],2)+pow(next_goal[1]-next_robot_position[1],2))
        
        if (result<10): 
            return 0.1
        if (result<30):
            return 0.2
        if (result<50):
            return 0.3
        if (result<70):
            return 0.4
        if (result<100):
            return 0.5
        else :
            return 0.6
    
    def F2_near_agents(self, state, action, mode):
        global one_foot
        robot_xyP = [0, 0]
        robot_xyP[0] = state[0] ## robot_xyP : action 이후 로봇의 위치
        robot_xyP[1] = state[1]

        if action == "UP": ## action이 UP 이면 로봇의 y좌표에 one_foot을 더함
            robot_xyP[1] += one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP) ##action 이후 로봇 반경 내 agents 수 구함
        elif action == "DOWN":
            robot_xyP[1] -= one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP)
        elif action == "RIGHT":
            robot_xyP[0] += one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP)
        elif action == "LEFT":
            robot_xyP[0] -= one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP)

        return NumberOfAgents * 0.1


    def reward_distance(self, state, action, mode):
        from model import space_connected_linear
        global SumList
        SumOfDistances = 0 ##agent 하나로부터 출구까지의 거리의 합
        floyd_distance = self.model.floyd_distance

        evacuation_points = [] ## 출구 찾기~
        if(self.model.is_left_exit): 
            evacuation_points.append(((0,0), (5, 45)))
        if(self.model.is_up_exit):
            evacuation_points.append(((0,45), (45, 49)))
        if(self.model.is_right_exit):
            evacuation_points.append(((45,5), (49, 49)))
        if(self.model.is_down_exit):
            evacuation_points.append(((5,0), (49, 5)))

        for i in self.model.agents: ##SumOfDistaces 구하는 과정
            if(i.dead == False and (i.type==0 or i.type==1)):
                agent_space = self.model.grid_to_space[int(round(i.xy[0]))][int(round(i.xy[1]))]
                
                next_goal = space_connected_linear(((agent_space[0][0],agent_space[0][1]), (agent_space[1][0], agent_space[1][1])), self.model.floyd_warshall()[0][((agent_space[0][0],agent_space[0][1]), (agent_space[1][0], agent_space[1][1]))][evacuation_points[0]])
                agent_space_x_center = (agent_space[0][0] + agent_space[1][0])/2
                agent_space_y_center = (agent_space[1][0] + agent_space[1][1])/2
                a = (floyd_distance[((agent_space[0][0],agent_space[0][1]), (agent_space[1][0], agent_space[1][1]))][evacuation_points[0]] 
                - math.sqrt(pow(agent_space_x_center-next_goal[0],2) + pow(agent_space_y_center-next_goal[1],2)) 
                + math.sqrt(pow(next_goal[0]-i.xy[0],2) + pow(next_goal[1]-i.xy[1],2)))
                
                ###준아야 너는 아래 코드를 수정해야 하며, 문제는 같은 space 내에서 agents가 움직이는 걸 반영하지 못하는 것에 있단다. 위 코드를 보며 수정하도록 야호^^
                # SumOfDistances += floyd_distance[(agent_space[0][0], agent_space[0][1]), (agent_space[1][0], agent_space[1][1])][evacuation_points[0]]
                SumOfDistances += a

        t = SumList[4]
        SumList[4] = SumList[3]
        SumList[3] = SumList[2]
        SumList[2] = SumList[1]
        SumList[1] = SumList[0]
        SumList[0] = SumOfDistances

        reward = (SumList[1]+SumList[2]+SumList[3]+SumList[4])/4 - SumOfDistances

        return reward
    
    def reward_difficulty_space(self,state,action,mode):
        global DifficultyList

        # gray spcce의 좌표만 가진 list 생성
        space_list = self.model.space_list #space list 저장
        room_list = self.model.room_list #room list 저장
        semi_safe_zone_list = [[[0, 0], [5, 45]], [[0, 45], [45, 49]], [[45, 5], [49, 49]], [[5, 0], [49, 5]]] # 후보 safe zone list 저장
        pure_gray_space = [] # safe zone이랑 room 빼서 순수 gray 만듦
        for sublist_a in space_list:
            if sublist_a not in room_list and sublist_a not in semi_safe_zone_list:
                pure_gray_space.append(sublist_a)
        #print('pure_space list 입니다:',pure_gray_space)

        
        
        # exit이 있는 공간 좌표 safe_zone_space으로 저장
        exit_coordinate = self.model.exit_rec
        #print('exit :',exit_coordinate)
        #print('exit #1 ',exit_coordinate[0][0],exit_coordinate[0][1])
        if exit_coordinate[0][0] >=0 and exit_coordinate[0][0]<=5 and exit_coordinate[0][1]>=0 and exit_coordinate[0][1] <= 45:
            safe_zone_space =  ((0,0),(5,45))
        elif exit_coordinate[0][0] >=0 and exit_coordinate[0][0]<=45 and exit_coordinate[0][1]>=45 and exit_coordinate[0][1] <= 49:
            safe_zone_space = ((0,45),(45,49))
        elif exit_coordinate[0][0] >=45 and exit_coordinate[0][0]<=49 and exit_coordinate[0][1]>=5 and exit_coordinate[0][1] <= 49:
            safe_zone_space =  ((45,5),(49,49))
        else:
            safe_zone_space = ((5,0),(49,5))

        
        each_space_agent_num = self.agents_in_each_space2() # 각 구역에 몇명있는지 저장
        shortest_distance = self.model.floyd_distance

        sum_Difficulty = 0 # 여기에 (출구로부터 회색 공간까지의 거리 * 그 공간의 agent 수) 들의 합을 저장함
        for sublist in pure_gray_space: # 순수 gray 공간에 agent 수 찾기~
            tuple_key = tuple(map(tuple, sublist))
            #print("그래이 공간과 그 공간의 사람 수",sublist,each_space_agent_num.get(tuple_key))
            #print("safe zone과 그레이 공간사이의 거리1",shortest_distance[safe_zone_space][tuple_key])
            gray_space_agent_mul_difficulty = shortest_distance[safe_zone_space][tuple_key] * each_space_agent_num.get(tuple_key)
            sum_Difficulty += gray_space_agent_mul_difficulty

        DifficultyList[4] = DifficultyList[3]
        DifficultyList[3] = DifficultyList[2]
        DifficultyList[2] = DifficultyList[1]
        DifficultyList[1] = DifficultyList[0]
        DifficultyList[0] = sum_Difficulty

        reward = (DifficultyList[1]+DifficultyList[2]+DifficultyList[3]+DifficultyList[4])/4 - sum_Difficulty
        return reward
    def select_Q(self, state) :
        global robot_xy
        global one_foot
        global NUMBER_OF_CELLS
        action_list = ["UP", "DOWN", "LEFT", "RIGHT"]
        r_x = robot_xy[0]
        r_y = robot_xy[1]
        
        del_object = []
        for k in action_list:
            if (k == "UP"):
                if(self.model.valid_space[int(round(r_x))][int(round(r_y+one_foot))]==0):
                    del_object.append("UP")
                    
            elif (k == "DOWN"):
                if(self.model.valid_space[int(round(r_x))][int(round(r_y-one_foot))]==0 or (r_y-one_foot)<0):
                    del_object.append("DOWN")

            elif (k == "LEFT"):
                if(self.model.valid_space[int(round(max(r_x-one_foot, 0)))][int(round(r_y))]==0 or (r_x-one_foot)<0):
                    del_object.append("LEFT")
            elif (k == "RIGHT"):
                if(self.model.valid_space[int(round(min(r_x+one_foot, NUMBER_OF_CELLS)))][int(round(r_y))]==0) :
                    del_object.append("RIGHT")
        del_object= list(set(del_object))
        for i in del_object:
            action_list.remove(i)

        Q_list = []
        for i in range(len(action_list)):
            Q_list.append(0)
        MAX_Q =-999999999
        ## 초기 selected 값 random 선택 ##
        values = ["UP", "DOWN", "LEFT", "RIGHT"]
        selected = random.choice(values)
        direction_agents_num = self.four_direction_compartment()
        #print(direction_agents_num)
        for j in range(len(action_list)):
            f1 = self.F1_distance(state, action_list[j], "GUIDE")
            f2 = self.F2_near_agents(state, action_list[j], "GUIDE")
            f3 = self.F3_direction_agents(state, action_list[j], "GUIDE", direction_agents_num)
            f4 = self.F4_difficulty_avg(state, action_list[j], "GUIDE", direction_agents_num) ###f4 test
            #print("\nf4 : ", f4, "action: ", action_list[j])
            #print(direction_agents_num)
            if True : # guide 모드일때 weight는 feature_weights_guide
                Q_list[j] = (f1 * self.feature_weights_guide[0] + f2 *self.feature_weights_guide[1] + f3 * self.feature_weights_guide[2])
            else :                           # not guide 모드일때 weight는 feature_weights_not_guide 
                Q_list[j] = (f1 * self.feature_weights_not_guide[0] + f2 * self.feature_weights_not_guide[1] + f3 * self.feature_weights_not_guide[2])
            
            if (Q_list[j]>MAX_Q):
                MAX_Q= Q_list[j]
                selected = action_list[j]
            exploration_rate = 0.05
            
            if random.random() <= exploration_rate:
                #actions = ["UP", "DOWN", "LEFT", "RIGHT"]
                selected = random.choice(action_list)
            
            self.now_action = [selected, "GUIDE"]
        return self.now_action


    def four_direction_compartment(self):
        from model import space_connected_linear
        #from model import Model 
        global robot_xy 
        global one_foot
        r_x = robot_xy[0]
        r_y = robot_xy[1]
        four_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        

        del_object = []
        for k in four_actions:
            if (k[0] == "UP"):
                if(self.model.valid_space[int(round(r_x))][int(round(r_y+one_foot))]==0):
                    del_object.append("UP")
                    
            elif (k[0] == "DOWN"):
                if(self.model.valid_space[int(round(r_x))][int(round(r_y-one_foot))]==0 or (r_y-one_foot)<0):
                    del_object.append("DOWN")

            elif (k[0] == "LEFT"):
                if(self.model.valid_space[int(round(max(r_x-one_foot, 0)))][int(round(r_y))]==0 or (r_x-one_foot)<0):
                    del_object.append("LEFT")
            elif (k[0] == "RIGHT"):
                if(self.model.valid_space[int(round(min(r_x+one_foot, NUMBER_OF_CELLS)))][int(round(r_y))]==0) :
                    del_object.append("RIGHT")
        
        del_object= list(set(del_object))
        for i in del_object:
            four_actions.remove([i])
            four_actions.remove([i])

        four_compartment = {}

        for j in four_actions:
            four_compartment[j] = []
        
        floyd_distance = self.model.floyd_distance 
        next_vertex_matrix = self.model.floyd_warshall()[0]

        now_s = self.model.grid_to_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]
        #print(robot_xy)
        #print(now_s)
        now_s = ((now_s[0][0], now_s[0][1]), (now_s[1][0], now_s[1][1]))
        now_s_x_center = (now_s[0][0] + now_s[1][0])/2
        now_s_y_center = (now_s[1][0] + now_s[1][1])/2 
        robot_position = [0, 0]
        robot_position[0] = robot_xy[0]
        robot_position[1] = robot_xy[1]
        only_space = []
        for sp in self.model.space_list:
            if (not sp in self.model.room_list and sp != [[0,0], [10, 10]] and sp != [[]]):
                only_space.append(sp)

        for i in only_space:
            key = ((i[0][0], i[0][1]), (i[1][0], i[1][1]))
            if(key==now_s):
                continue
            next_goal = space_connected_linear(now_s, next_vertex_matrix[now_s][key])
            original_distance = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-robot_position[0],2)+pow(next_goal[1]-robot_position[1],2))
            up_direction = 99999
            down_direction = 99999
            left_direction = 99999
            right_direction = 99999

            for m in four_actions:
                if (m=="UP"):
                    up_direction = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-robot_position[0],2)+pow(next_goal[1]-(robot_position[1]+one_foot),2))        
                elif (m=="DOWN"):
                    down_direction = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-robot_position[0],2)+pow(next_goal[1]-(robot_position[1]-one_foot),2))        
                elif (m=="LEFT"):
                    left_direction = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-(robot_position[0]-one_foot),2)+pow(next_goal[1]-robot_position[1],2))     
                elif (m=="RIGHT"):
                    right_direction = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-(robot_position[0]+one_foot),2)+pow(next_goal[1]-robot_position[1],2))     

            min = up_direction 
            min_direction = "UP"

            if(min>down_direction):
                min_direction = "DOWN"
                min = down_direction
            if(min>left_direction):
                min_direction = "LEFT"
                min = left_direction
            if(min>right_direction):
                min_direction = "RIGHT"
                min = right_direction  

            four_compartment[min_direction].append(i)
            #print("\nfour compartment : ", four_compartment, "\n")
        return four_compartment
    
    def F3_direction_agents(self, state, action, mode, compartment_direction):
        sum = 0 
        each_space_agents_num = self.agents_in_each_space2()
        for i in compartment_direction[action]:
            key = ((i[0][0], i[0][1]), (i[1][0], i[1][1]))
            sum += each_space_agents_num[key]
        return sum * 0.1
    
    def F4_difficulty_avg(self, state, action, mode, compartment_direction): # 가까워지는(action 했을 때) 구역의 난이도 평균 return
        ## 가까워지는 구역이 없으면 return 0, 가까워지는 구역에 출구가 포함되어 있으면 출구 제외 난이도 평균 (출구는 난이도 -1)
        a = []
        for val in compartment_direction[action]: # action을 했을 때 가까워지는 구역
            if val != list(map(list, self.model.exit_compartment)): #출구 포함되어 있으면 제외
                a.append(self.model.dict_NoC[tuple(map(tuple, val))])
        if len(a) != 0 :
            return np.mean(a)
        else:
            return 0
        


    def calculate_Guide_Q(self,state): # state 집어 넣으면 max_Q 내주는 함수
        global robot_xy
        global one_foot
        action_list = ["UP", "DOWN", "LEFT", "RIGHT"]
        r_x = robot_xy[0]
        r_y = robot_xy[1]
        
        del_object = []
        for k in action_list:
            if (k == "UP"):
                if(self.model.valid_space[int(round(r_x))][int(round(r_y+one_foot))]==0):
                    del_object.append("UP")
                    
            elif (k == "DOWN"):
                if(self.model.valid_space[int(round(r_x))][int(round(r_y-one_foot))]==0 or (r_y-one_foot)<0):
                    del_object.append("DOWN")

            elif (k == "LEFT"):
                if(self.model.valid_space[int(round(max(r_x-one_foot, 0)))][int(round(r_y))]==0 or (r_x-one_foot)<0):
                    del_object.append("LEFT")
            elif (k == "RIGHT"):
                if(self.model.valid_space[int(round(min(r_x+one_foot, NUMBER_OF_CELLS)))][int(round(r_y))]==0) :
                    del_object.append("RIGHT")
        del_object= list(set(del_object))
        for i in del_object:
            action_list.remove(i)
            action_list.remove(i)

        Q_list = []
        for i in range(len(action_list)):
            Q_list.append(0)
        MAX_Q = -9999999

        direction_agents_num = self.four_direction_compartment()
        for j in range(len(action_list)):
            f1 = self.F1_distance(state, action_list[j], "GUIDE")
            f2 = self.F2_near_agents(state, action_list[j], "GUIDE")
            f3 = self.F3_direction_agents(state, action_list[j], "GUIDE", direction_agents_num)
            

            Q_list[j] = (f1 * self.feature_weights_guide[0] + f2 * self.feature_weights_guide[1] + f3 * self.feature_weights_guide[2])
            
            if (Q_list[j]>MAX_Q):
                MAX_Q= Q_list[j]
        return MAX_Q

    def calculate_Max_Q(self,state): # state 집어 넣으면 max_Q 내주는 함수
        global robot_xy
        global one_foot
        action_list = [["UP", "GUIDE"], ["UP", "NOGUIDE"], ["DOWN", "GUIDE"], ["DOWN", "NOGUIDE"], ["LEFT", "GUIDE"], ["LEFT", "NOGUIDE"], ["RIGHT", "GUIDE"], ["RIGHT", "NOGUIDE"]]
        
        r_x = robot_xy[0]
        r_y = robot_xy[1]
        
        del_object = []
        for k in action_list:
            if (k[0] == "UP"):
                if(self.model.valid_space[int(round(r_x))][int(round(r_y+one_foot))]==0):
                    del_object.append("UP")
                    
            elif (k[0] == "DOWN"):
                if(self.model.valid_space[int(round(r_x))][int(round(r_y-one_foot))]==0 or (r_y-one_foot)<0):
                    del_object.append("DOWN")

            elif (k[0] == "LEFT"):
                if(self.model.valid_space[int(round(max(r_x-one_foot, 0)))][int(round(r_y))]==0 or (r_x-one_foot)<0):
                    del_object.append("LEFT")
            elif (k[0] == "RIGHT"):
                if(self.model.valid_space[int(round(min(r_x+one_foot, NUMBER_OF_CELLS)))][int(round(r_y))]==0) :
                    del_object.append("RIGHT")
        del_object= list(set(del_object))
        for i in del_object:
            action_list.remove([i, "GUIDE"])
            action_list.remove([i, "NOGUIDE"])

        Q_list = []
        for i in range(len(action_list)):
            Q_list.append(0)
        MAX_Q = -9999999

        direction_agents_num = self.four_direction_compartment()
        for j in range(len(action_list)):
            f1 = self.F1_distance(state, action_list[j][0], action_list[j][1])
            f2 = self.F2_near_agents(state, action_list[j][0], action_list[j][1])
            f3 = self.F3_direction_agents(state, action_list[j][0], action_list[j][1], direction_agents_num)
            
            if action_list[j][1] == "GUIDE": # guide 모드일때 weight는 feature_weights_guide
                Q_list[j] = (f1 * self.feature_weights_guide[0] + f2 * self.feature_weights_guide[1] + f3 * self.feature_weights_guide[2])
            else :                           # not guide 모드일때 weight는 feature_weights_not_guide 
                Q_list[j] = (f1 * self.feature_weights_not_guide[0] + f2 * self.feature_weights_not_guide[1] + f3 * self.feature_weights_not_guide[2])
            
            if (Q_list[j]>MAX_Q):
                MAX_Q= Q_list[j]
        return MAX_Q
    
    def calculate_Q(self, state, action):
        global robot_xy
        
        f1 = self.F1_distance(state, action[0], action[1])
        f2 = self.F2_near_agents(state, action[0], action[1])
        direction_agents_num = self.four_direction_compartment()
        f3 = self.F3_direction_agents(state, action[0], action[1], direction_agents_num)
        Q= 0
        if(action[1] == "GUIDE"):
            Q = f1 * self.feature_weights_guide[0] + f2*self.feature_weights_guide[1] + f3 * self.feature_weights_guide[2]
        else:
            Q = f1 * self.feature_weights_not_guide[0] + f2*self.feature_weights_not_guide[1] + f3*self.feature_weights_not_guide[2]

        return Q

    def update_weight(self,reward):  
        global robot_xy

        alpha = 0.1
        discount_factor = 0.1
        next_robot_xy = [0,0]
        next_robot_xy[0] = robot_xy[0]
        next_robot_xy[1] = robot_xy[1]
        
        # select_Q에서 내주는 action에 따라 다음 state 계산
        if self.now_action[0] == 'UP':
            next_robot_xy[1] += 1
        elif self.now_action[0] == 'DOWN':
            next_robot_xy[1] -= 1
        elif self.now_action[0] == 'RIGHT':
            next_robot_xy[0] += 1
        else:
            next_robot_xy[0] -= 1

        #print('잘됐나',self.select_Q(robot_xy)[0],robot_xy,next_robot_xy)

        # 현재 state와 다음 state의 max_Q 값 계산
        next_state_max_Q = self.calculate_Max_Q(next_robot_xy)
        present_state_Q = self.calculate_Q(robot_xy, self.now_action)
        #print('good?',next_state_max_Q,present_state_Q)

        # 여기부터 실제 업데이트 진행
        direction_agents_num = self.four_direction_compartment()
        
        f1 = self.F1_distance(robot_xy, self.now_action[0], self.now_action[1])
        f2 = self.F2_near_agents(robot_xy, self.now_action[0], self.now_action[1])
        f3 = self.F3_direction_agents(robot_xy, self.now_action[0], self.now_action[1],direction_agents_num)
        
        
        
        selected_action = self.now_action[1]
        #print('weight :',self.feature_weights_guide,self.feature_weights_not_guide)
        #print('select_Q :', self.now_action)
        if selected_action == "GUIDE":
            ### 아래는 얘네 각각 출력..
            #print("//self.w1(", self.w1, ") += alpha(0.1) * (reward(", reward, ") + discount_factor(0.2) * next_state_max_Q(", next_state_max_Q, ") - present_state_Q(", present_state_Q, ")) * f1(", f1, ")")
            self.w1 += alpha * (reward + discount_factor * next_state_max_Q - present_state_Q) * f1
            self.w2 += alpha * (reward + discount_factor * next_state_max_Q - present_state_Q) * f2
            self.w3 += alpha * (reward + discount_factor * next_state_max_Q - present_state_Q) * f3
            self.feature_weights_guide[0] = self.w1 #계산한 w1, w2를 리스트에 재할당해야 반영됨
            self.feature_weights_guide[1] = self.w2

        if selected_action == "NOGUIDE":
            self.w4 += alpha * (reward + discount_factor * next_state_max_Q - present_state_Q) * f1
            self.w5 += alpha * (reward + discount_factor * next_state_max_Q - present_state_Q) * f2
            self.w6 += alpha * (reward + discount_factor * next_state_max_Q - present_state_Q) * f3
        
        return
