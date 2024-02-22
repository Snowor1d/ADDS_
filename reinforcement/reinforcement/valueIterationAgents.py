# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount #시간에 대한 감소(깜마)
        self.iterations = iterations #반복수 
        self.values = util.Counter() # A Counter is a dict with default 0  
        #self.values 
        #self.values[state] = 가치값이 나온다 ~ 
        self.runValueIteration()    
            
    def runValueIteration(self): # 미리 설정한 state1->action->state2 reward를 바탕으로 self.values라는 table를 채운 것 
        # self.values라는 table안에는 각 state에 대한 value값이 들어있는 듯 ~~~ 
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        for i in range (self.iterations):
            values=self.values.copy()
            for s in self.mdp.getStates(): # s는 모든 state가 돌아가면서 배정 
                self.values[s] = -float('inf')
                for a in  self.mdp.getPossibleActions(s): # each state s에 대한 action들
                    temp=0
                    for sprime,prob in self.mdp.getTransitionStatesAndProbs(s,a): #state s에서 action a를 했을 때
                        temp+=prob*(self.mdp.getReward(s, a, sprime)+ self.discount *values[sprime])
                    self.values[s] = max(self.values[s], temp)
                if self.values[s] == -float('inf'):
                    self.values[s] = 0.0



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        temp=0
        for sprime, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # 그 state에서 action를 하면 나오는 ~ 
            # getTran~ -> [(state1, state1_P), (state2, stat2_P), (state3, stat3_P)] 
            temp += prob * (self.mdp.getReward(state, action, sprime) + self.discount * self.values[sprime])
            #getReward -> state, action 했을때 그 다음의 sprime라는 state로 넘어갔을때의 가치를 도출해내는 함수 
            #sprime -> state 
            #prob -> 그 state로 갈 확률 
        return  temp
    def computeActionFromValues(self, state): #어떤 state에서 가장 큰 reward를 갖다주는 action을 도출해내는 함수
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state): # 종료 condition? 
            return None
        maxnum, Action = -float('inf'), None
        for a in self.mdp.getPossibleActions(state): #그 state에서 어떤 action을 할 수 있는지 나옴 
            temp = self.computeQValueFromValues(state, a) #가치 계산  
            if temp > maxnum: 
                maxnum, Action = temp, a
        return Action

    def getPolicy(self, state): #computeaction과 같은 함수 
        return self.computeActionFromValues(state)

    def getAction(self, state): #computeaction과 같은 함수 
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def computeQValues(self, state):
        # Returns a counter containing all qValues from a given state

        actions = self.mdp.getPossibleActions(state)  # All possible actions from a state
        qValues = util.Counter()  # A counter holding (action, qValue) pairs

        for action in actions:
            # Putting the calculated Q value for the given action into my counter
            qValues[action] = self.computeQValueFromValues(state, action)

        return qValues

    def runValueIteration(self):

        for k in range(self.iterations):

            state = self.mdp.getStates()[k %  len(self.mdp.getStates())]
            best = self.computeActionFromValues(state)
            if best is None:
                V = 0
            else:
                V = self.computeQValueFromValues(state, best)
            self.values[state] = V


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    def runValueIteration(self):
        allStates = self.mdp.getStates()
        predecessors = dict()
        for state in allStates:
            predecessors[state]=set()
        for state in allStates:
            allactions=self.mdp.getPossibleActions(state)
            for a in allactions:
                possibleNextStates = self.mdp.getTransitionStatesAndProbs(state, a)
                for nextState,pred in possibleNextStates:
                    if pred>0:
                        predecessors[nextState].add(state)
        pq = util.PriorityQueue()
        for state in allStates:

            stateQValues = self.computeQValues(state)

            if len(stateQValues) > 0:
                maxQValue = stateQValues[stateQValues.argMax()]
                diff = abs(self.values[state] - maxQValue)
                pq.push(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                return;
            state = pq.pop()
            stateQValues = self.computeQValues(state)
            maxQValue = stateQValues[stateQValues.argMax()]
            self.values[state] = maxQValue
            for p in predecessors[state]:

                pQValues = self.computeQValues(p)
                maxQValue = pQValues[pQValues.argMax()]
                diff = abs(self.values[p] - maxQValue)

                if diff > self.theta:
                    pq.update(p, -diff)