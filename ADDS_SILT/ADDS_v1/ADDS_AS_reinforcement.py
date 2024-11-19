from mesa import Model
from mesa import Agent
from agent import FightingAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import os 
import agent
from agent import WallAgent
import random
import copy
import math
import numpy as np
import os
import time

import agent
import model
import time
import sys

#-------------------------#
visualization_mode = 'off' # choose your visualization mode 'on / off
run_iteration = 1500
number_of_agents = 30 # agents 수
#-------------------------#

for j in range(run_iteration):
    print(f"{j} 번째 학습 ")
    result = []
    the_number_of_model = 0

    for model_num in range(5):
        step_num = 0
        
        # 모델 생성 및 실행에 실패하면 반복해서 다시 시도
        while True:
            try:
                # model 객체 생성
                model_o = model.FightingModel(number_of_agents, 70, 70, model_num)
                the_number_of_model += 1
                print("------------------------------")
                print(f"{the_number_of_model}번째 학습")
                break  # 객체가 성공적으로 생성되면 루프 탈출
            except Exception as e:
                print(e)
                print("error 발생, 다시 시작합니다")
                continue  # 모델 생성에 실패하면 다시 시도
        
        # 모델이 성공적으로 생성되었으므로 step 진행
        while True:
            try:
                step_num += 1
                model_o.step()
                
                # 모델이 99% 탈출 조건에 도달하면 루프 종료
                if model_o.alived_agents() == 1:
                    break
            except Exception as e:
                print(e)
                print("error 발생, 다시 시작합니다")
                # step 수행 중 오류가 발생하면, model 생성부터 다시 시작
                break

        print("99% 탈출에 걸리는 step : ", step_num)