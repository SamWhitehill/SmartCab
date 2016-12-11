import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import os
from collections import namedtuple

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def fGetQStateTable(self):
        #return Table of states and actions, which contains rewards under each 
        #action for each state
        
        #State = namedtuple('State', 'Light	OncomingCar 	LeftCar	RightCar \
        #                       NextWaypoint')

        #st1 = State(Light='Red', OncomingCar=None,LeftCar=None,	RightCar='forward',
         #                      NextWaypoint='forward',	Action_None=None,Action_Left=None,Action_Right=-1,
          #      Action_Forward=2)
            
        seqDict={}
        lstDirections=[None,'left','right','forward']
        lstOncoming =[None,'forward']
        lstActions=[0,0,0,0]  # assume actions are none, left, right and forward in this order.
        for lt in ['red','green']:
            for nwyp in lstDirections:
                for oncoming in lstOncoming:
                    for leftcar in lstOncoming:
                        for rightcar in lstOncoming:
                            state = self.StateTuple(Light=lt, OncomingCar=oncoming,LeftCar=leftcar,	RightCar=rightcar,
                                           NextWaypoint=nwyp)

                    	        #Action_None=0,Action_Left=0,Action_Right=0,
                                #        Action_Forward=0

                            seqDict[state]=[0,0,0,0]

            


        #lstQMatrix=[]
        return seqDict

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.gamma =.15
        self.alpha =.7
        #epsilon must decay over each episode, to get less random over time.
        self.epsilon =.5

        self.StateTuple = namedtuple('State', 'Light	OncomingCar 	LeftCar	RightCar \
                               NextWaypoint')

        #use this accessing action columns in Q matrix by action name
        #instead of hard coded index values
        self.ActionIndexDict ={None:0,'left':1,'right':2,'forward':3}

        #get initial Q-table , to be iteratively updated...
        self.QTable =self.fGetQStateTable()

        self.state=None
        self.lastReward=None
        self.lastAction=None
        self.episode =-1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state=None
        self.lastReward=None
        self.lastAction=None
        self.timestep =1 #used for updating learning rate: alpha
        #for each new trip, add 1 to episode, this is used in decyaing learning and epsilon
        self.episode =self.episode +1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        #cannot update first time this is called as we need a previous state.
        if self.state !=None:
            #update Q table using previous state, next state ,reward earned and action
            #that took us to this state

            sLeftCar=self.fGetOtherCarLocation(inputs['left'])
            sRightCar=self.fGetOtherCarLocation(inputs['right'])
            sOncomingCar=self.fGetOtherCarLocation(inputs['oncoming'])

            self.updateQTable(self.state,self.StateTuple(Light=inputs['light'], OncomingCar=sOncomingCar,
                               LeftCar=sLeftCar,	RightCar=sRightCar,
                               NextWaypoint=self.next_waypoint),self.lastReward,
                               self.lastAction,self.episode)
            
            #update timestep, in order to update learning rate
            self.timestep=self.timestep+1

        #5 rows, 7 cols grid.
        sLeftCar=self.fGetOtherCarLocation(inputs['left'])
        sRightCar=self.fGetOtherCarLocation(inputs['right'])
        sOncomingCar=self.fGetOtherCarLocation(inputs['oncoming'])

        # TODO: Update state
        self.state =self.StateTuple(Light=inputs['light'], OncomingCar=sOncomingCar,
                               LeftCar=sLeftCar,	RightCar=sRightCar,
                               NextWaypoint=self.next_waypoint)
        
        # TODO: Select action according to your policy
        #action = 'left'
        action =self.fGetNextAction(self.episode)
        lDblRand =action[1]
        action =action[0]
        self.lastAction =action
        # Execute action and get reward
        reward = self.env.act(self, action)

        #per udacity dev. it is correct to have zero reward for NOT moving at green light.
        #if action==None and reward==0 and inputs['light']=='green':
        #    print('zero reward stopped at green light')
        
        #store reward last earned as we need it for next update in updating
        #Q table (i.e., need reward earned on last step )
        self.lastReward=reward
        #best action is not always  self.next_waypoint,
        # b/c u could be AT A RED LIGHT.  
        #print (self.env.act(self, 'right'), self.env.act(self, 'left'), self.env.act(self, 'forward'))

        # TODO: Learn policy based on state, action, reward

        #if inputs['oncoming']!=None or inputs['left']!=None or inputs['right']!=None :
         #   print(inputs)
               #value of left: is direction that car is going (about to go) on your left
               #value of right: is direction that car is about to go on you right
        if self.episode>15:
            if reward <0 and (lDblRand>self.epsilon/float(self.epsilon+self.episode)):
                #figure out why reward is negative at this point
                print( action, lDblRand)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, nextWayPoint={}".format(deadline, inputs, action, reward,    self.next_waypoint)  # [debug]

    def updateQTable(self, pOldStateTuple,pNewStateTuple,pReward,pAction,pEpisodeNum):
        #update Q-table based on curent state, current reward,
        #discounted reward from next state chosen using best action.
        #pState - StateTuple object( defined in class init) used for accessing
        #Qtable row
        #pReward- reward just earned
        #pAction taken.

        try:
            lActionIndex =self.ActionIndexDict[pAction]

            #get max reward of next state action pair.
            lMaxQ=max(self.QTable[pNewStateTuple])

            lAlpha =self.alpha/float(self.alpha+pEpisodeNum)
            if pEpisodeNum==0:
                #1st time episode is 0, so we do not divide to avoid creating alpha=1.
                lAlpha =self.alpha
            #probability of going to this next state  is 1 - epsilon.
            #lProbability =1 #- self.epsilon
      
            #Q[s,a] ?(1-?) Q[s,a] + ?(r+ ?maxa' Q[s',a'])
            lUpdateValue =(1-lAlpha)*self.QTable[pOldStateTuple][lActionIndex]
            lUpdateValue =lUpdateValue+ lAlpha*(pReward +self.gamma*lMaxQ)

            #problem here, need to find out why it logs a positive reward
            # for taking different action than next waypoint
            if pOldStateTuple.NextWaypoint!=pAction and pAction !=None and lUpdateValue>0:
                print('problem logging reward')

            #update the record as identified by state
            self.QTable[pOldStateTuple][lActionIndex] =lUpdateValue
        
        except Exception as e:
            print(e)



    def fGetNextAction(self,pEpisodeNum):
        #return next action to take
        #this could be random or policy based.
        #get maximum action for this state
        try:
            listActions =self.QTable[self.state]
            lMaxQ=max(listActions)
            lIndex =listActions.index(lMaxQ)
            dictActions =self.ActionIndexDict
            action = dictActions.keys()[dictActions.values().index(lIndex)]
            #may need to look at probability here too ?
        
            #decay the epsilon rate over time
            lEpsilon =self.epsilon/float(self.epsilon+pEpisodeNum)
            if pEpisodeNum==0:
                lEpsilon =self.epsilon

            lDblRand =random.random()
            #only take optimal action with probability 1-epsilon, otherwise random.
            if lDblRand<lEpsilon:
                action =random.choice([None, 'forward', 'left', 'right']) 
        
        except Exception as e:
            print(e)     
                    
        return action , lDblRand
       
    def fGetOtherCarLocation(self,pStrCarLocation):
        #parse car location such that it's either None (no car) or 'forward'
        # not sure of all possible car directions (e.g, forward, right,etc..)
        # but it appears all that matters is either there is another car on my right or NOT
        # on my left or not, oncoming or not
        lResult=None
        if pStrCarLocation!=None:
            lResult='forward'
        
        return lResult
        

def run():
    """Run the agent for a finite number of trials."""
    print('remove the chdir statement when submitting to udacity')
    os.chdir( "C:\\Udacity\\NanoDegree\\Smart Cab Project\\smartcab" )

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    #e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    #per 1st task ,set enforce_deadline=False
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_test=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
