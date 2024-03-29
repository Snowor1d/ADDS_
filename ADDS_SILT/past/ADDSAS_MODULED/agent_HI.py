from mesa import Agent
import math
import numpy as np
import random
import copy

num_remained_agent = 0

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
robot_radius = 20 #로봇 반경 -> 10미터 
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


def space_connected_linear(xy1, xy2):
    check_connection = []
    for i1 in range(51):
        tmp = []
        for j1 in range(51):
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

    def step(self) -> None: # 각 step마다 agent가 어떤 행동을 할지에 대한 내용이 적혀있음
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
        if(self.type != 3): #robot은 죽지 않는다  # 탈출구에 가면 agent가 죽는 코드
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
            now_stage = ((0,0), (5, 45))
        return now_stage

    def which_goal_agent_want(self):
        

        if(self.goal_init == 0):
            now_stage = self.check_stage_agent() 
            goal_candiate = self.model.space_goal_dict[now_stage] #1개에서 4개까지 있겠지 #space goal dict는 space넣으면 extend까지한 goal의 좌표가 나옴
            if(len(goal_candiate)==1):
                goal_index = 0
            else:
                goal_index = random.randint(0, len(goal_candiate)-1)
            self.now_goal = goal_candiate[goal_index]
            self.goal_init = 1
            self.previous_stage = now_stage
        now_stage = self.check_stage_agent() #now_stage -> agent가 현재 어느 space에 있는가 
        if(self.previous_stage != self.check_stage_agent()): # 다음 space로 넘어온 상태
            goal_candiate = self.model.space_goal_dict[now_stage] # ex) [[2,0], [3,5],[4,1]] 
            goal_candiate2 = []
            if(len(goal_candiate)>1): # 여기부터가 이전 space로 다시 돌아가지는 않게 하는 장치
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

        if (self.type == 3): # type이 3이면 로봇
            new_position = self.robot_policy2() # robot_policy2는 로봇이 행동하는 코드
            self.model.grid.move_agent(self, new_position)
            return

        new_position = self.test_modeling() # test_modeling은 일반 agent가 행동하는 코드
        if(self.type ==0 or self.type==1):
            self.model.grid.move_agent(self, new_position) ## 그 위치로 이동
    
    def robot_policy(self): # 이거는 무시해도됨
        time_step = 0.2
        from model_HI import Model
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
                evacuation_points.append(((0,0), (5, 45)))
            if(self.model.is_up_exit):
                evacuation_points.append(((0,45), (45, 49)))
            if(self.model.is_right_exit):
                evacuation_points.append(((45,5), (49, 49)))
            if(self.model.is_down_exit):
                evacuation_points.append(((5,0), (49, 5))) #evacuation_points에 탈출구들 저장 

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
        #print("현재 골 : ", self.robot_now_path[self.robot_waypoint_index])
        robot_goal = self.robot_now_path[self.robot_waypoint_index]
        

        d = (pow(self.robot_now_path[self.robot_waypoint_index][0]-robot_xy[0],2) + pow(self.robot_now_path[self.robot_waypoint_index][1]-robot_xy[1],2)) #현재 위치와 goal까지의 거리 구하기
        if (d<3):
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
        if(next_x>49):
            next_x = 49
        if(next_y>49):
            next_y = 49

        return (next_x, next_y)
    
    def robot_policy2(self): # 어딘가에 들리고 safe 존으로 가는 걸 반복
        time_step = 0.2
        from model_HI import Model
        global random_disperse
        global robot_status
        global robot_xy 
        global robot_radius
        global robot_ringing
        global robot_goal
        global past_target
        self.drag = 1
        robot_status = 1
        space_agent_num = self.agents_in_each_space() #어느 stage에 몇명이 있는가
        floyd_distance = self.model.floyd_distance # floyd_distance[stage1][stage2] = 최단거리 
        floyd_path = self.model.floyd_path #floyd_path[stage1][stage2] = stage_x 
                                           #floyd_path[stage_x][stage_2] = stage_y 
                                           #floyd_path[stage_y][stage_2] = ste... 
                                           # s1 -> sx -> sy -> .. stage
        
    
        

        self.robot_space = self.model.grid_to_space[int(robot_xy[0])][int(robot_xy[1])] #로봇이 어느 space에 있는지 나온다 

        if(self.mission_complete == 1): #새로운 탈출 path를 찾는다 # safe zone에 들렀다.
            self.robot_now_path = [] # [[1,3], [4,5], [5,1]] 
            agent_max = 0 #agent가 가장 많은 stage 
            self.find_target(space_agent_num, floyd_distance) # 공간의 꼭짓점 좌표가 나옴

            past_target = self.save_target # 전에 갔던 공간을 가지 않기 위해 past_target으로 저장해둔다

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
            self.drag = 1 # 사람들이 이끌리게 만드는 term 
        else:
            robot_status = 0
            self.drag = 0
        print("현재 골 : ", self.robot_now_path[self.robot_waypoint_index])
        print(self.robot_waypoint_index)
        print(robot_xy)
        print(self.save_target)
        print(self.robot_now_path)
        robot_goal = self.robot_now_path[self.robot_waypoint_index]
        
        d = (pow(self.robot_now_path[self.robot_waypoint_index][0]-robot_xy[0],2) + pow(self.robot_now_path[self.robot_waypoint_index][1]-robot_xy[1],2)) #현재 위치와 goal까지의 거리 구하기
        if (d<3):  # d는 실제 좌표의 거리  이게 작아지면 인덱스 하나 크게 해서 다음 목표로
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
            desired_speed = 10
        else:
            desired_speed = 10

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
                    repulsive_force[0] += 5*k*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 5*k*np.exp(-(d/2))*(d_y/d)
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
        if(next_x>49):
            next_x = 49
        if(next_y>49):
            next_y = 49
        #print(F_x, F_y)
            
        #if(self.dead != True):
            #print(self.now_goal)
            #print(desired_force[0], desired_force[1])
            #print(F_x, F_y)

        #self.robot_guide = 0
        return (next_x, next_y)

    def agents_in_each_space(self):
        global num_remained_agent
        from model_HI import Model
        space_agent_num = {}
        for i in self.model.space_list:
            space_agent_num[((i[0][0],i[0][1]), (i[1][0], i[1][1]))] = 0
        for i in self.model.agents:
            space_xy = self.model.grid_to_space[int((i.xy)[0])][int((i.xy)[1])]
            if(i.dead == False and (i.type==0 or i.type==1)):
                space_agent_num[((space_xy[0][0], space_xy[0][1]), (space_xy[1][0], space_xy[1][1]))] +=1 
        for j in space_agent_num.keys():
            print(j, "공간에 ", space_agent_num[j], "명이 있음")
            num_remained_agent += space_agent_num[j]
            #print('agents_in_each_space function employed')
        return space_agent_num

    def find_target(self, space_agent_num, floyd_distance):
        global past_target
        self.robot_now_path = []
        agent_max = 0
        space_priority = {}
        distance_to_safe = {}
        

        evacuation_points = []
        if(self.model.is_left_exit): 
            evacuation_points.append(((0,0), (5, 45)))
        if(self.model.is_up_exit):
            evacuation_points.append(((0,45), (45, 49)))
        if(self.model.is_right_exit):
            evacuation_points.append(((45,5), (49, 49)))
        if(self.model.is_down_exit):
            evacuation_points.append(((5,0), (49, 5))) #evacuation_points에 탈출구들 저장 

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

        #print("distance_to_safe :", distance_to_safe)
                
        
        for l in space_agent_num.keys():
            space_priority[l] = distance_to_safe[l] * space_agent_num[l]
            if(l==past_target):
                space_priority[l] -= 10000
        #(space_priority)
        agent_max = 0
        print(space_priority)
        for k in space_priority.keys():
            if (space_priority[k]>agent_max):
                self.save_target = k
                agent_max = space_priority[self.save_target]
        min_distance = 1000
        for m in evacuation_points: #space_target에서 가장 가까운 탈출구를 찾기 
            if(floyd_distance[self.save_target][m]<min_distance):
                self.save_point = m
                min_distance = floyd_distance[self.save_target]
        print(self.save_target)
        
    def test_modeling(self):
        global robot_radius
        global robot_xy
        global robot_status
        from model_HI import Model
        global random_disperse

        x = int(round(self.xy[0]))
        y = int(round(self.xy[1]))
        #temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
        temp_loc = [(x-2, y), (x-1, y), (x+1, y), (x+2, y), (x, y+1), (x, y+2), (x, y-1), (x, y-2), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)] # 주변 12칸
        near_loc = []
        for i in temp_loc:
            if(i[0]>0 and i[1]>0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
                near_loc.append(i) # near은 실제로 현실적인 주변 좌표고.
        near_agents_list = []
        for i in near_loc:
            near_agents = self.model.grid.get_cell_list_contents([i]) # 이걸 하면 near에 있는 fighting agent 객체가 나옴
            if len(near_agents):
                for near_agent in near_agents:
                    near_agents_list.append(near_agent) #kinetic 모델과 동일 # 결국 near_agents_list에는 주변 그리드의 agent가 쌓임

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
                continue    # d가 3보다 크면 이 아래가 안 돌고 다음 for문이 돌아.

            
            # print("F : ", F)
            # if(d>0 and near_agent.dead == False):
            #     F_x += (F*(d_x/d))
            #     F_y += (F*(d_y/d))
            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽 / 회색 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0

                elif(near_agent.type == 1): ## agents
                    repulsive_force[0] += 1/4*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 1/4*np.exp(-(d/2))*(d_y/d) # 뒤에 거는 벡터고, 앞에 거가 힘의 크기다.

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

        robot_x = robot_xy[0] - self.xy[0]
        robot_y = robot_xy[1] - self.xy[1]
        robot_d = math.sqrt(pow(robot_x,2)+pow(robot_y,2))
        if(robot_d<robot_radius and robot_status == 1):
            goal_x = robot_x
            goal_y = robot_y
            goal_d = robot_d
            self.type = 1 # 가이드 당하고 있는 agent는 type=1
            self.now_goal = robot_goal
            
        else :
            self.type = 0


        if(goal_d != 0):
          desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :                      # intend_force랑 desired_speed는 상수
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
        #print(F_x, F_y)
            
        #if(self.dead != True):
            #print(self.now_goal)
            #print(desired_force[0], desired_force[1])
            #print(F_x, F_y)

        self.robot_guide = 0
        return (next_x, next_y)