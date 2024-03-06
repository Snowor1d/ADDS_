# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

#qlearningAgents -> Question2까지는 value iteration 돌릴 때 경험으로부터 학습하는게 아닌 미리 만들어진 MDP 모델 (미리 정의된 가치값들)을 썼잖아??
#그러나 MDP 모델을 정의하는 것은 현실에서는 불가능.. 경험으로부터 얻어야 하는 것
#learning from trial & error

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvalues = util.Counter()

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalues[(state,action)]
        util.raiseNotDefined()

    def computeValueFromQValues(self, state): # 행동 가치로부터 state의 가치 계산 
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state) # state로 부터 action을 얻는 것 같은데 legal이 뭐지?? 
        out = -float('inf') 
        for a in actions: # actions 중에 가장 좋은 action을 찾는 듯 ~ 
            if self.getQValue(state,a)>=out:
                out = self.getQValue(state, a)
        
        return out if out!=(-float('inf'))else 0.0
        util.raiseNotDefined()

    def computeActionFromQValues(self, state): 
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None. 어느 state에서 best action을 찾는 거래 
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        out = -float('inf')

        for a in actions: #actions 중에서 가장 가치가 높은 action의 가치를 찾는다 
            if self.getQValue(state, a) > out:
                out = self.getQValue(state, a)
                outa = a

        lis = []
        for a in actions: 
          if self.getQValue(state, a) == out: #가장 높은 가치를 가지는 action들을 list에 집어 넣는다  
              lis.append(a)
        if out != -float('inf'):
            outa=random.choice(lis) #이것 중 하나 선택 ~ 
        else:
            outa=None 
        return outa 
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # random action or best policy action 
        # Pick Action
        legalActions = self.getLegalActions(state) 
        if len(legalActions)==0:
            return None 
        if util.flipCoin(self.epsilon): # self.epsilon?? random action을 고를 지 best policy를 고를 지 선택하게 하는 것 
            # greedy algorithm과 epsilon greedy algorithm이 있음. greedy algorithm은 미래를 생각하지 않고 각 단계에서 가장 최선의 선택을 하는 기법. 
            # greedy algorithm은 exploration이 충분히 이루어지지 않았기 때문에 최상의 결과가 나올 것이라 확신할 수 없음 
            # 이에 대한 보완책이 epsilon-greedy algorithm. 일정한 확률로 랜덤 선택을 해서 exploration을 하게 함 
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qvalues[(state,action)] = (1-self.alpha) * (self.getQValue(state,action)) + (self.alpha) * (reward+self.discount*self.computeValueFromQValues(nextState))
        #util.raiseNotDefined()
        #self.alpha는 learningrate?? 얼마나 빨리 학습시킬지 정하는 건가?
        #맞는 듯~ self.alpha가 클 수록 새로운 경험의 반영이 많이 됨 
        #어떤 state에서 어떤 action을 했을 때의 가치는 = 원래 Q 가치 + (새로운 reward + 다음 state로 갈때의 가치 * 시간 감쇠) .. 로 업데이트 
        #Q-learning 공식 참조 
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent): #QLearningAgent 상속 
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent): 
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    # approximate Q-learning은 Q(s,a) 테이블을 모두 저장하지 않음. 현실과 비슷한 환경일 수록 변수가 많아지기 때문에 테이블을 모두 저장하는 것은 비효율적 
    # 테이블을 모두 저장하는 대신, 이전의 경험을 기반으로 현재의 상황을 일반화하여 해석 ('Feature'와 'Weight' 사용 )
    # pacman게임에서 feature이라 한다면, 가장 가까운 ghost와의 거리, 가장 가까운 food와의 거리, ghost의 수, 남은 food의 수 ~ 가 될 수 있을 것 
    # weight는 각 feature에 대한 가중치 - 해당 특성이 얼마나 중요한가? 
    # linear value function을 통해 가중치를 활용하여 Q(s,a), V(s)를 도출할 수 있다고 함 
    # Q-learning이 Q_v 테이블을 업데이트 한다면, Approximate Q-learning은 가중치를 업데이트 함
    # ㅋㅋㅋ 

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter() # features에 대한 가중치 

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector 
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        w= self.getWeights()
        featureVector = self.featExtractor.getFeatures(state,action) #features 추출 

        return w *featureVector 
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward: float): #학습하면서 가중치 업데이트 
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        featureVector = self.featExtractor.getFeatures(state,action) #현 state에서 action에 대한 features를 추출하고..
        
        maxQFromNextState = self.computeValueFromQValues(nextState) # 다음 state에 대한 가치를 가져온다 
        actionQValue = self.getQValue(state,action) # 또한 행동에 대한 가치도 가져온다 

        for feature in featureVector: # 이것들을 이용해서 feature에 대한 가중치를 업데이트 한다 
            self.weights[feature] += self.alpha * (reward + self.discount * maxQFromNextState - actionQValue) * \
                                     featureVector[feature]

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
