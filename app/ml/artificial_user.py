

class World:
    def __init__(self):
        pass

    def get_state(self):
        pass

    def execute_action(self,a):
        pass

class GameTree:
    class GameNode:
        def __init__(self, state, action, par):
            self.state = state
            self.action = action
            self.visited = 0.0
            self.par = par
            # key = the goal aka desired result, val = the times that goal was reached through this node
            self.goal_outcomes = dict()
            # key = state of the system in which this node can be accessed, val = list<GameNode>
            self.children = dict()

        def probability(self):
            """The likelihood of ending up in this state"""
            return self.visited / self.par.visited

        def probability_for(self, goal):
            """The likelihood that the given GameNode will lead the 'player' to their final goal"""
            return self.goal_outcomes.get(goal,0.0) / self.visited

        def get_utility(self, goal, policy_params):
            """Returns the utility that the 'player' will receive if they pick this action"""
            pass

        def apply_action(self, world):
            self.visited += 1.0
            world.execute_action(self.action)

        def update(self, goal, win=0):
            """
            Updates the number of times that this node led to the desired goal
            :param goal: the goal that was supposed to be reached
            :param win: 1 if goal was reached 0 otherwise
            :return: 
            """
            self.goal_outcomes[goal] = self.goal_outcomes.get(goal) + win
            self.par.update(goal, win)

        def prune(self):
            pass

        def get_next_move(self, state, goal, policy_params):
            """
            Returns the best action for a certain goal given the current state of the world
            :param state: the current state of the world
            :param goal: the desired final outcome
            :param policy_params: additional data regarding the choice of a next action
            :return: GameNode with highest utility
            """
            if goal in self.children:
                return self.children.get(goal)[0]
            best = None
            max_u = 0
            nodes = self.children.get(state)
            for n in nodes:
                u = n.get_utility(goal, policy_params)
                if u > max_u:
                    max_u = u
                    best = n
            return best

    def __init__(self, policy):
        self.root = None
        self.cursor = None
        self.policy = policy

    def get_next_action(self, state, goal):
        if not self.cursor:
            self.cursor = self.root
        new_c = self.cursor.get_next_move(state, goal,self.policy)
        self.cursor = new_c
        return new_c


class AIUser:
    def __init__(self, world):
        """
        Creates a new AI user
        :param world: World instance wrapper for the environment  
        """
        self.world = world
        self.game_tree = GameTree(None)

    def learn(self):
        pass

    def execute(self, goal):
        state = self.world.get_state()
        while state != goal:
            a = self.game_tree.get_next_action(state, goal)
            a.apply_action(self.world)
            state = self.world.get_state()
