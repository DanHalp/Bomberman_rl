import json
import os
import numpy as np
from time import sleep, time
from settings import e as e
from settings import s as s
import sys
INVALID_ACTION = 6
DROP_BOMB = 4
KILLED_SELF = 13
GOT_KILLED = 14


def write_dict_to_file(agent):
    """ In order to use a 'trained' agent, it is necessary to write it's Q_table into a file.
     For that we use the "JSON" method. Our Q_talbe is a DICTIONARY, and JSON knows how to
     deal with it. JSON DOES NOT know how to work with np.arrays. """
    with open("q_table.json", "w") as output:
        output.write(json.dumps(agent.q_table))
        output.close()


def read_dict_from_file(agent):
    """Load a ready Q_table dictionary to our agent. """
    with open("q_table.json", "r") as input:
        agent.q_table = json.load(input)
        input.close()


def training_radius(agent):
    """When training with just coins, we need to set radius according to coins and not crates. """
    coins = np.array(agent.game_state["coins"])
    self = np.array([agent.game_state["self"][0], agent.game_state["self"][1]])
    agent.radius = max(min(np.linalg.norm(self - coins, axis=1)), 4 * np.sqrt(2))


def real_radius(agent):
    """When a trained agent plays, he'll calculate the searching radius according to the  next crate."""
    crates = np.array(np.where(agent.game_state["arena"] == 1))
    crates = crates.reshape(len(crates[0]), 2)
    self = np.array([agent.game_state["self"][0], agent.game_state["self"][1]])
    agent.radius = max(min(np.linalg.norm(self - crates, axis=1)), 4 * np.sqrt(2))


def stateToStr(state):
    """ Turns a state representation of list to a string. The strings is used as keys for the
     q_table dictionary. """
    return ", ".join(state)


def bombs_without_walls(agent, pos):
    """
    Even if a bomb is less than 3 tiles away, if there is a bomb between the agent and the bomb,
    the blast does not put the agent in dagner.
    :param agent:
    :param pos: The coordinate of the tile, for which we check it it is in danger.
    :return: a list of bombs in real danger.
    """

    # Bombs = only the bombs that stand in a straight line from the agent.
    bombs = np.array([[x, y] for x, y, t in agent.game_state["bombs"] if
                      (x == pos[0] or y == pos[1])])
    if len(bombs):
        bombs_dis_check = np.linalg.norm(bombs - pos, axis=1)
        bombs = bombs[bombs_dis_check < 4]  # Only relevant bombs.
        if len(bombs):
            bombs_check = bombs_dis_check[bombs_dis_check < 4]  # The dis from relevant bombs.
            temp = np.sign(bombs - pos)  # This way we determine in which directions the bombs are.
            indices = []  # What bombs are actual dangerous for us?
            for i, direction in enumerate(temp):  # For each direction.
                temp_pos, clearFlag = pos, False  # clearFlag is False if we found a wall
                #  - meaning the bomb would not be added to the relevant bombs.
                for _ in range(int(bombs_check[i])):
                    temp_pos = np.array([temp_pos[0] + direction[0], temp_pos[1] + direction[1]])
                    if agent.game_state["arena"][temp_pos[0], temp_pos[1]] == -1:
                        # we found a wall between the bomb and the agent.
                        clearFlag = True
                        break
                if not clearFlag:
                    # We are in danger from that bomb.
                    indices.append(i)
            return bombs[np.array(indices)] if len(indices) else []
    return []


def find_priority(agent, curr_state, cand_name):
    """
    In the cases where there is no specific action that needs to be done (like collecting coins or dropping a bomb
    on an enemy), we want to determine what is the best action to make. Waiting in our place without any danger is
    useless, and therefore we will choose only among the movements actions. It is done by computing a score for each
    possibility, and choosing the best option.

    Score is calculated by: the sum of scores for each element: crate, coin or enemy. For each one we shall
    calculate the distance of all the instances from the position, where we would be, if we went in that direction.
    Each element has a weight (crate:1, coins:80, enemy:1), and we will divide the each instance's self score by
    it's relative distance from the checked position, squared. Meaning = score: for each element:
    #instances * W(element) / (relative_distance ** 2)
    :param curr_state: current built state
    :param cand_name: what should we look for in the current state. For a simple movement, we should look for "empty",
            but when we drop a bomb, we would look for "danger1", to determine what is the best route to escape.
    :return:
    """

    candidates = np.array(curr_state)[:4]
    candidates = np.where(candidates == cand_name)[0]

    if not len(candidates):
        return

    # Calculate scores, as described above:
    scores, best_indices = np.zeros(len(candidates)), []
    for i, index in enumerate(candidates):
        col, row = agent.changes_point_in_dir[index]
        temp_pos = np.array([agent.curr_pos[0] + col, agent.curr_pos[1] + row])

        if cand_name == "danger":
            bombs_cord = bombs_without_walls(agent, temp_pos)
            if len(bombs_cord):
                dis = np.linalg.norm(temp_pos - bombs_cord, axis=1)
                closest_bomb = bombs_cord[np.argmin(dis)]
                if min(dis) < np.linalg.norm(agent.curr_pos - closest_bomb):
                    scores[i] = np.inf
                    continue
                else:
                    scores[i] += len(bombs_cord) * -20
                    bombs = np.array(agent.game_state["bombs"])[:, [0,1]]
                    dis = np.linalg.norm(temp_pos - bombs)
                    scores[i] += np.sum(-20 * (1/dis))

        # curr_tile = np.array([self_coordinates[0] + col, self_coordinates[1] + row])
        for object_name in agent.weights.keys():
            if object_name == "arena":
                objects = np.array(np.where(agent.game_state["arena"] == 1))
                objects = objects.transpose()

            elif object_name == "coins":
                objects = np.array(agent.game_state["coins"])
            else:
                enemies = np.array(agent.game_state["others"])
                objects = enemies[:, [0, 1]].astype(int) if len(enemies) else []

            if len(objects):
                distance = np.linalg.norm(temp_pos - objects, axis=1)
                indices = np.where(distance <= agent.radius)[0]
                rel_distance = distance[indices]
                if len(rel_distance):
                    temp = np.ones(len(rel_distance)) * agent.weights[object_name]
                    scores[i] += np.sum(temp / ((rel_distance + 1) ** 2))


    # Choose the best route, according to the scores.
    if cand_name == "danger":
        scores = np.array(scores)
        candidates = candidates[scores != np.inf]
        scores = scores[scores != np.inf]
        if len(scores) == 0:
            return
        best_indices = candidates[np.where(np.array(scores) == max(scores))[0]]
    else:
        best_indices = candidates[np.where(np.array(scores) == max(scores))[0]]

    j = np.random.choice(best_indices)
    curr_state[j] = "priority"



def isInDanger(agent, pos):
    """
    Finds out if the tile "pos" is in "danger", which means - It is within a blowing distance of a bomb.
    It calculates the distances from all bombs, and for those which are in within 3 steps (A bomb's explosion radius)
    it looks for those who are located in a straight line (x1-x2 == 0 or y1-y2 == 0) with "pos". A fitting bomb puts us
    in danger.
    :param pos: checked tile.
    :param without_self_bomb: When used to determine if a path is secured from a self dropped bomb, pos is of course on
           a straight line from our bomb, and we try to look if other bombs threat us.
    :return: Boolean. True for danger and False for nothing to worry about.
    """

    bombs = bombs_without_walls(agent, pos)
    if len(bombs):
        bombs_dis_self = np.linalg.norm(bombs - pos, axis=1)
        return len(np.where(bombs_dis_self < 4)[0]) > 0
    return False


def clearPath(agent, index, curr_pos, counter=4):
    """
    In case we haved dropped a bomb - we would like to choose the best route away from it, in order not to blow up along
    with it. For that, for each possible route (an "empty" tile) we look further ahead for 4 tiles in the same
    direction. We do that recursively. If some of them has a way out (a direction not on the line from the bomb),
    it is a "clear" path. It prevents the agent to go into a too short dead end
    (for the agents always looks only to the next tile), and die, trapped between walls, crates it can not blow out
     of its way and a ticking bomb.
    :param index: index of direction we look into (0:left, 1:up, 2:right, 3:down)
    :param dir_var: each direction has fixed coefficients, according to board logic. E.g - direction "left" has
            dir_var = [-1,0] for we need to go -1 steps on the columns and 0 on the rows.
    :param curr_pos: current tile we determine for, in the path.
    :param counter: number of tile left to check in path.
    :return: is it a clear path?
    """
    try:
        dir_var = agent.changes_point_in_dir[index]
        new_pos = np.array([curr_pos[0] + dir_var[0], curr_pos[1] + dir_var[1]])
        if counter == 0:
            return True
        elif agent.game_state["arena"][new_pos[0], new_pos[1]] != 0:  # If the tile is not empty to move into.
            return False
        elif agent.game_state["explosions"][new_pos[0], new_pos[1]] > 2:  # Is it safe from sparse fire?
            return False
        elif len(agent.game_state["others"]):
            for enemy in agent.game_state["others"]:
                if enemy[0] == new_pos[0] and enemy[1] == new_pos[1]:
                    return False

        # Look for an escape rout. Lets say the bomb is to our right - is there a clear way up or down?
        for direction in [0,1,2,3]:
            temp_dir_var = agent.changes_point_in_dir[direction]
            if np.abs(direction - index) != 2 and direction != index and \
                    agent.game_state["arena"][new_pos[0] + temp_dir_var[0], new_pos[1] + temp_dir_var[1]] == 0:
                return True

        # Check next tile in path.
        return clearPath(agent, index, new_pos, counter - 1)
    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()

        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Reward_update: ", e, exc_type, fname, exc_tb.tb_lineno)
        bul = 5


def setup(agent):
    """ Some useful objects and values for our agent. """

    agent.number_of_games = 0
    agent.curr_state = None  # The most used argument. Saves the last created state.
    agent.prev_state = None  # Used for the Q_function.
    agent.episode_rewards_log = dict()

    agent.relevant_actions = [0,1,2,3,4,6,7,14]
    agent.movements = {0:0, 1:2, 2:1, 3:3, 7:4, 4:5, 6:6,14:14}  # Translate setting.e to our purpose.
    agent.rewards = {"bomb": -70, "danger1": -40, "coin": 80, "priority": 60, "enemy": -10,
                     "wall": -70, "empty": -10, "danger2": -30, "danger3": -20,"invalid": -70, "goal_action": 80,
                     "dead":-70, "danger":-40, "crate":-70}
    agent.changes_point_in_dir = {"left":[-1,0], "up":[0,-1], "right":[1,0],
                                  "down":[0,1], 0:[-1,0], 1:[0,-1], 2:[1,0], 3:[0,1]}
    agent.actions = ["LEFT","UP","RIGHT","DOWN","BOMB","WAIT"]  # Actions in the needed order.
    agent.train_flag = False  # Todo - check what is that for.

    # The epsilon arguments are meant for the epsilon-greedy policy.
    # The 'gamma' and 'alpha' values are hyperparameters.
    agent.epsilon = 1.0
    agent.epsilon_min = 0.20  # Relatively high minimum eps would prevent overfitting
    agent.epsilon_decay = 0.9995
    agent.gamma = 0.65
    agent.alpha = 0.25

    # Radius is a variable used for looking for priorities near by, which attracts our agents,
    # in case a tile of state is "empty". Weights are the priority of each object.
    agent.radius = None
    agent.radius_incrementer = 0.1  # Each turn, radius is incremented by 0.05
    agent.weights = {"arena":1, "coins":30, "others":1}  # Weights for each object on the board.

    # Load the q_table:
    try:  # If the q_table.json file doesn't exit yet.
        read_dict_from_file(agent)
    except Exception as ex:
        agent.q_table = dict()


def act(agent):
    """ This method is used by the game engine, and here we determine what shall action be made
    next.
    If NOT in TRAINING MODE, we find what state our agent is in, and then reach for the
    fitting state in Q_table, and take the np.argmax of all values in the 'knowledge_list'.
    This index correlates with the fitting string action in agent.actions.
    If we ARE in TRAINING MODE, we used the epsilon-greedy policy, which helps us to balance the
    concept of EXPLORING and EXPLOITING. """

    agent.logger.info('Pick action according to pressed key')

    # Working with e-greedy policy:
    # Figure radius
    if agent.game_state["step"] == 1 and agent.radius is None:
        training_radius(agent) if s.crate_density == 0 else real_radius(agent)
    # First step: find the current state.
    # Todo: For report, timings of finding a state.
    start = time()
    agent.curr_state = find_state(agent)
    end = time() - start
    # Second step:
    string = stateToStr(agent.curr_state)

    if s.gui:
        action = np.argmax(agent.q_table[string])
    elif agent.train_flag:  # This flag is our own flag, that acknowledge us that this is training.
        action_rewards = np.array(agent.q_table[string])
        if 0 in action_rewards:
            # Todo - Write about that in paper.
            indices = np.where(action_rewards == 0)[0]
            action = np.random.choice(indices)

        #  Epsilon-Greedy Policy: After every game that is finished while training, we update our
        #  epslion, that is initially set to 1. After about 3000 games, it reaches the epsilon_min
        #  value. The point is, that we want the agent to play as it should, but still try random
        #  things for the states it reaches, to keep learning. This policy is called
        #  "Exploration-Exploitation".
        elif np.random.uniform(0,1) < agent.epsilon:
            # In order to speedup training, if there are many undone actions for a specific state,
            # we shall make the agent do an action it has not done yet, on purpose.
            action = np.random.randint(0,6)
        else:
            action = np.argmax(agent.q_table[string])

    # When not in training mode, we should use the pre-prepared Q table.
    else:
        # Todo - For the paper, here we take all choices with the best score and choose randomly
        # todo  - between them
        try:
            # The agent is sent to the fight after about 15,000 training games. But still,
            # it is possible for it to get into a state, which is not known to it - and in that case
            # an exception would rise. In that case, we choose a random action, and continue
            # playing.
            knowledge_list = np.array(agent.q_table[stateToStr(agent.curr_state)])
            best_indices = np.where(knowledge_list == max(knowledge_list))[0]
            action = np.random.choice(best_indices)
        except KeyError as ke:
            action = np.random.randint(0,6)

    # Set action and change radius.
    agent.radius += agent.radius_incrementer
    agent.next_action = agent.actions[action]


def find_state(agent):
    """ This is the main method of this project. We defined a state as a tuple:
    (left, up, right, down, self, own_bomb). First we look around us, and define the tiles around us.
    If a neighbor is full, it's easy - put it's content as term in the state. If it's empty, we
    should look around and try to determine what attracts us the most (It's gonna be improved!)  """

    # First step: A state is a tuple, made of the 4 directions around our agent
    # (left, up, right, down) and (self, self_bomb).
    # In each iteration we determine our state in the specific
    # direction, rather there is a danger of explosion or a pre-exploded bomb, a tile,
    # an actor, a coin etc.

    options = [(-1,0,"left"), (0,-1,"up"), (1,0,"right"), (0,1,"down"), (0,0,"self")]
    curr_state = list()
    self_coordinates = np.array([agent.game_state["self"][0], agent.game_state["self"][1]])
    agent.curr_pos = self_coordinates
    for col, row, direction in options:
        # Which tile we are looking at:
        checked_coordinates = np.array([self_coordinates[0] + col, self_coordinates[1] + row])

        # Let's check what is found in "checked_coordinates". If it is full with something,
        # we shall check what it is, and then insert it to "curr_state" list, and continue to
        # the next iteration. If it is empty we shall try and determine what is the best option
        # for the agent to be attracted to, or avoid, in the current direction.

        # first, let's check whether the tile is full or not:

        if agent.game_state["arena"][checked_coordinates[0], checked_coordinates[1]] == -1:
            curr_state.append("wall")
            continue
        elif agent.game_state["arena"][checked_coordinates[0], checked_coordinates[1]] == 1:
            curr_state.append("crate")
            continue
        elif agent.game_state["explosions"][checked_coordinates[0], checked_coordinates[1]] > 1:
            curr_state.append("wall")
            continue

        players = np.array(agent.game_state["others"])
        if len(players):
            players_dis = np.linalg.norm(players[:, [0, 1]].astype(int) - checked_coordinates, axis=1)
            if len(players[players_dis == 0]):
                curr_state.append("enemy")
                continue

        bombs = np.array([[x, y] for x, y, t in agent.game_state["bombs"] if
                          (x == checked_coordinates[0] or y == checked_coordinates[1])])
        if len(bombs):
            bombs_dis_check = np.linalg.norm(bombs - checked_coordinates, axis=1)
            if len(bombs[bombs_dis_check == 0]) > 0:
                curr_state.append("bomb")
                continue

            if isInDanger(agent, checked_coordinates):
                curr_state.append("danger")
                continue

        if direction == "self":
            curr_state.append("empty")
            continue

        # Or.. Is it a coin?
        coins = np.array(agent.game_state["coins"])
        if len(coins):
            coins = np.linalg.norm(coins - checked_coordinates, axis=1)
            if len(coins[coins == 0]) > 0:
                curr_state.append("coin")
                continue

        # Otherwise, the tile is empty. Let us try and understand what danger or rewards are
        # nearby.

        # Maybe there's another player?
        if len(players):
            players_dis = np.linalg.norm(players[:, [0, 1]].astype(int) - checked_coordinates, axis=1)
            if len(players[players_dis == 0]) > 0:
                curr_state.append("enemy")
                continue

        # If none of the above fitted - this tile is empty, and shall tested after the loop.
        curr_state.append("empty")

    # Second step:
    # Priority: If there is no imminent danger, we will try and the best way to move.
    # For that we will consider all objects on the board (except to walls) and grade
    # each "empty" direction by the number of objects and their weights, where:
    # crates = 1, coins = 80, enemy = 1, and by their relevant distance from the agent.
    # so a direction's value would be : #object * W(object) / dis_from_agent
    if curr_state[4] == "bomb":
        empty = np.where(np.array(curr_state)[:4] == "danger")[0]
        for j in empty:
            if not clearPath(agent, j, agent.curr_pos):
                curr_state[j] = "wall"

    if not ("enemy" in curr_state[:4] and agent.game_state["self"][3] == 1) and "coin" not in curr_state[:4]:
        # If droping a bomb will stuck us in a dead end. In this case there is not priority.
        if "crate" in curr_state[:4] and agent.game_state["self"][3]:
            empty = np.array(curr_state)[:4]
            empty = np.where(empty == "empty")[0]
            for j in empty:
                if not clearPath(agent, j, agent.curr_pos):
                    curr_state[j] = "wall"

        # If there's an empty spot, it should be a candidate to be the priority direction.
        elif "empty" in curr_state[:4]:
            find_priority(agent, curr_state, "empty")

        # If there is no empty tile around us, but there is danger, we should try and find out
        # If we still should move into it, another one, or wait in our place.
        elif "danger" in curr_state[:4]:
            find_priority(agent, curr_state, "danger")

    # This is a boolean value, if the agent has a self-bomb already
    curr_state.append(str(agent.game_state["self"][3] == 0))

    # If the state is not in q_table, add it with a list of 6 0's.
    string_state = stateToStr(curr_state)
    try:
        agent.q_table[string_state]
    except Exception as ex:
        agent.q_table[string_state] = list(np.zeros(6))

    return curr_state


def reward_update(agent):
    """Called once per step to allow intermediate rewards based on game events.
    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.
    """

    # School code:
    agent.logger.debug(f'Encountered {len(agent.events)} game event(s)')
    # Our code:
    # Restarting episode log.
    if agent.game_state["step"] == 1:
        # agent.epsilon = 1.0
        agent.episode_rewards_log = dict()
        agent.train_flag = True
        training_radius(agent) if s.crate_density == 0 else real_radius(agent)

    # This means nothing had happened on the board, meaning no action has been made.
    if len(agent.events) == 0:
        return

    agent.curr_pos = np.array([agent.game_state["self"][0], agent.game_state["self"][1]])
    state_string = stateToStr(agent.curr_state)
    # First step: We created a list of relevant actions from given settings.e
    action, Rt = 0, 0
    for event in agent.events:
        # if we have made a move:
        if event not in agent.relevant_actions:
            continue

        # Translate the games settings into our own:
        action = agent.actions.index(agent.next_action)

        if event == GOT_KILLED:
            Rt += agent.rewards["dead"]

        # An "invalid action": an invalid penalty is -20 points.
        # Invalid actions : Walk into wall, crate or drop a bomb when the agent has none.
        if event == INVALID_ACTION:
            action = agent.actions.index(agent.next_action)
            if agent.curr_state[action] == "priority":
                Rt += agent.rewards[agent.curr_state[action]]
            else:
                Rt += agent.rewards["invalid"]

        # Movement on board
        if action < DROP_BOMB:   # action is in [0,1,2,3]
            curr_action = agent.curr_state[action]
            Rt += agent.rewards[curr_action]

        # Drop Bomb
        elif action == DROP_BOMB:  # action == 4
            if "crate" in agent.curr_state[:4] or "enemy" in agent.curr_state[:4]:
                Rt += agent.rewards["goal_action"]

        # Wait
        else:
            Rt += -5 if agent.curr_state[4] != "danger" else agent.rewards["bomb"]

        # We update the previous state:
        # This is the implementation of the q-learning algorithm.
    try:
        # The formula of the Q-function relies on the values of the next state. Because the states
        # function is not deterministic, we do not know what the next state is. Therefore, we save
        # the current values, and only at the next step we will use them to calculate this stats's
        # reward, once we have the values of the next one.
        prev_state_string = stateToStr(agent.prev_state)
        a, g = agent.alpha, agent.gamma
        Q0sa0 = agent.q_table[prev_state_string][agent.prev_action]
        Qs1a1 = agent.rewards["dead"] if GOT_KILLED == agent.events[-1] else max(agent.q_table[state_string])
        curr_reward = Q0sa0 + a * (agent.prev_reward + g * Qs1a1 - Q0sa0)
        agent.q_table[prev_state_string][agent.prev_action] = curr_reward

    except Exception as ex:
        # If an exception is raised, it is because that we are at the beginning of the game,
        # And only one step has been made, so there is no actual "previous state".
        pass

    # Update current values, for the next state.
    agent.prev_action = action
    agent.prev_reward = Rt
    agent.prev_state = agent.curr_state


def end_of_episode(agent):
    """Called at the end of each game to hand out final rewards and do training.
    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """

    # school code:
    agent.logger.debug(f'Encountered {len(agent.events)} game event(s) in final step')
    # Our code:
    # Update reward for the last step. As always, we need the next state in order to calc the
    #  reward.
    reward_update(agent)
    reward_update(agent)

    # Reduce epsilon
    if agent.epsilon >= agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    # For each episode - we restart the Q function variables.
    agent.curr_state = None
    agent.prev_state = None
    agent.prev_action = None
    # Save the q_table, for later-use.
    write_dict_to_file(agent)
    agent.number_of_games += 1  # Used for debugging purposes.

    if agent.number_of_games % 50 == 0:
        print("#games = {}".format(agent.number_of_games))

