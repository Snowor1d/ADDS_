import agent_HI
import model_HI
import time

from agent_HI import num_remained_agent



model = model_HI.FightingModel(5,50,50)

n = 500  # n을 반복하려는 횟수로 설정


num_escaped_episodes = {
    "50%": 0,
    "80%": 0,
    "100%": 0
}
start_time = time.time()
for i in range(n): # 에피소드 n번 돌린다
    model.step()
    print('에피소드 수',i+1)
    
    #print('남은 agent 수', num_remained_agent)
    print('num_remained_agent',num_remained_agent)
#-----------------------------------------------------------------------------------------------------------------------
    if i == 0: # 처음 생성된 agent 수 저장
        num_assigned_agent = num_remained_agent

    if num_remained_agent <= int(num_assigned_agent*0.5): # 50% 이상 빠져나가면 그때 에피소드 수 저장
        if num_escaped_episodes["50%"] == 0:
            num_escaped_episodes["50%"] = i+1

    if num_remained_agent <= int(num_assigned_agent*0.2): # 80% 이상 빠져나가면 그때 에피소드 수 저장
        if num_escaped_episodes["80%"] == 0:
            num_escaped_episodes["80%"] = i+1
    
    if num_remained_agent == 0: # 모두 빠져나가면 그때 에피소드 수 저장 , 텍스트 파일에 저장
        if num_escaped_episodes["100%"] == 0:
            num_escaped_episodes["100%"] = i+1
            print(num_escaped_episodes)
            with open("example.txt", "a") as f:
                f.write("{}, {}, {}\n".format(num_escaped_episodes["50%"], num_escaped_episodes["80%"], num_escaped_episodes["100%"]))

        break
    else:
        num_remained_agent = 0 # 초기화
#-----------------------------------------------------------------------------------------------------------------------

end_time = time.time()
execution_time = end_time - start_time
print("코드 실행 시간:", execution_time, "초")