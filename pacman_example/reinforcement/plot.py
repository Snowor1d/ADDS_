import matplotlib.pyplot as plt

# 파일에서 값을 읽어 리스트로 저장하는 함수
def read_rewards(file_path):
    rewards = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                rewards.append(float(line.strip()))
            except ValueError:
                # 라인에 숫자가 아닌 값이 있을 경우 무시합니다.
                continue
    return rewards

# 그래프를 그리는 함수
def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker='o')
    plt.title('Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

# 파일 경로
file_path = 'reward.txt'

# 파일에서 값 읽기
rewards = read_rewards(file_path)

# 그래프 그리기
plot_rewards(rewards)
