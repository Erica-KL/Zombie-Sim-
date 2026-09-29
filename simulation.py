import math 
import random 

import db 

COLLISION_RADIUS = 12 #how close it has to be to touch
PERCEPTION_RADIUS = 120 #how far out can an agent see
TURN_RATE = 0.6 #the speed on which they turn 
            

CELL_SIZE = PERCEPTION_RADIUS #the cells size is the perception radius or 120
                      

DAY_LENGTH = 10 #one day 
NIGHT_START = DAY_LENGTH // 2 #5 ticks day 5 ticks night
TURN_TICKS = 3 * DAY_LENGTH #takes 3 days to turn 
SUSCEPTIBILITY_DECAY = 5 #decay by 5 points
EXPOSED_TICKS = 2 #A person is exposed for 2 ticks 
ATTACK_SCALE = 0.75 #less damage

#--------------Lab Stuff--------------------------------------

LAB_SIZE = 100 #
LAB_BOUNDS = (db.WIDTH / 2 - LAB_SIZE / 2, db.HEIGHT / 2 - LAB_SIZE / 2, 
              db.WIDTH / 2 + LAB_SIZE / 2, db.HEIGHT / 2 + LAB_SIZE / 2) 
LAB_POINT = {"x": db.WIDTH / 2, "y": db.HEIGHT / 2} 
INTELLIGENCE_THRESHOLD = 5 #have to have at least a 5 for research progress
RESEARCH_TICKS = 20 #ticks needed for progress to produce medicine
MEDICINE_SEARCH_RADIUS = 400 #because only a few infected exist on the map the carriers can search more area 
MEDICINE_POTENCY_MIN = 10 
MEDICINE_POTENCY_MAX = 20 
MEDICINE_HEAL = 25 #health is restored max health 100

def day_number(tick): #day increments  count 
    return tick // DAY_LENGTH + 1 #add one to the counter

def is_night(tick): 
    return tick % DAY_LENGTH >= NIGHT_START #start night during later of the half

def init_state(agents): 
    for agent in agents: 
        agent["heading"] = random.uniform(0, 2 * math.pi) #direction
        agent["status_timer"] = 0 #status timer vital for temp status like infected of exposed 
        agent["fighting"] = False #fight or flight
        agent["bites_survived"] = 0 #counts of failed bites 
        agent["lab_progress"] = 0 #tick progress made by labs
        agent["medicine_count"] = 0 #doses currently being carried 
        
#_-----------------------fight or flight-------------------

def decide_stances(agents):
    grid = _build_grid(agents) #fixed square agents are assigned 
    for agent in agents:
        if agent["agent_type"] != "human" or agent["health_status"] == "dead":
            continue #dead people dont do anything
        target = _nearest_of_type(agent, grid, "zombie") #Agent of interest nearby? No?
        agent["fighting"] = target is not None and random.uniform(0, 100) < agent["aggression"] #no target nothing happens if target roll random compare against agression

#---------------------------Infection progress-----------------

def advance_infections(agents):

    for agent in agents:
        if agent["health_status"] != "infected": #when someone is infected:
            continue 
        agent["status_timer"] += 1 #timer counts down (up) to turned 
        if agent["status_timer"] >= TURN_TICKS:
            agent["agent_type"] = "zombie" #flips agent type 
            agent["health_status"] = "turned" #now turned 
            agent["movement_speed"] = db.ZOMBIE_SPEED #speed decreases from human to zombie
            agent["status_timer"] = 0 #infection effect expires now turned 

#-------------------_Exposed-------------------------
def advance_exposure(agents):
    for agent in agents:
        if agent["health_status"] != "exposed": #for exposed humans 
            continue 
        agent["status_timer"] += 1 #status effect expriation goes down (up)
        if agent["status_timer"] >= EXPOSED_TICKS:
            agent["health_status"] = "healthy" #back to healthy
            agent["status_timer"] = 0 #reset 

#--------------------research progress ---------

def advance_research(agents, tick):
    produced = 0 #doses produced 
    for agent in agents:
        if (agent["agent_type"] != "human" or agent["health_status"] not in ("healthy", "exposed")
                or agent["medicine_count"] > 0 #if researcher carying a dose research will not progress 
                or agent["intelligence"] <= INTELLIGENCE_THRESHOLD or not _in_lab(agent)): #inteligence at 5 or above 
            continue 
        agent["lab_progress"] += 1 #dose progress 
        if agent["lab_progress"] >= RESEARCH_TICKS:
            agent["medicine_count"] += 1 #dose is done added to "inventory"
            agent["lab_progress"] = 0 #reset 
            produced += 1 #add to counter 
            print(f"tick {tick}: human #{agent['agent_id']} finishes a dose of medicine")
    return produced

def deliver_medicine(agents, tick):
    carriers = [a for a in agents if a["agent_type"] == "human" and a["medicine_count"] > 0] #anyone holding a dose
    patients = [a for a in agents if a["agent_type"] == "human" and a["health_status"] == "infected"] #who needs one
    for carrier in carriers:
        for patient in patients: #brute force oof though theres less agents with doses than total 
            if math.hypot(carrier["x"] - patient["x"], carrier["y"] - patient["y"]) <= COLLISION_RADIUS:
                patient["health_status"] = "healthy" #cure 
                patient["status_timer"] = 0 #how long it takes to turn healthy 
                drop = random.uniform(MEDICINE_POTENCY_MIN, MEDICINE_POTENCY_MAX) #random amount, 10-20 suceptibility drop
                patient["susceptibility"] = max(0, patient["susceptibility"] - drop) #suceptibility is a non zero
                patient["life_value"] = min(100, patient["life_value"] + MEDICINE_HEAL) #susceptibility is not above 100
                carrier["medicine_count"] -= 1 #dose used up
                print(f"tick {tick}: human #{carrier['agent_id']} cures human #{patient['agent_id']} "
                      f"-- susceptibility now {patient['susceptibility']:.1f}%, "
                      f"life now {patient['life_value']:.1f}, {carrier['medicine_count']} doses left")
                break #one dose from researcher per tick

def step(agents, dt=1.0, night=False):
    grid = _build_grid(agents) #new call
    for agent in agents:
        if agent["health_status"] == "dead":
            continue #dead agents stay put
        if night and agent["agent_type"] == "human":
            continue #asleep
        if agent["agent_type"] == "zombie":
            target, fleeing, bounds = _nearest_of_type(agent, grid, "human"), False, None #zombies always chase
        else:
            target, fleeing, bounds = _human_target(agent, grid, agents) #humans have the full priority list
        step_speed = agent["movement_speed"] * dt 
        if target:
            _move_away(agent, target, step_speed) if fleeing else _move_toward(agent, target, step_speed)
        else:
            _wander(agent, step_speed, dt) #wander 
        x0, y0, x1, y1 = bounds if bounds else (0, 0, db.WIDTH, db.HEIGHT) #bounds 
        agent["x"] = _clamp(agent["x"], x0, x1) #agents stuck within graph
        agent["y"] = _clamp(agent["y"], y0, y1)

def check_collisions(agents, tick, dt=1.0):
    grid = _build_grid(agents) #grid within block 
    for human in agents:
        if human["agent_type"] != "human" or human["health_status"] == "dead":
            continue #dead people cant fight 
        for other in _nearby(human, grid): #9 cell vision 
            if other["agent_type"] != "zombie" or other["health_status"] == "dead":
                continue #only living zombies count
            if math.hypot(human["x"] - other["x"], human["y"] - other["y"]) <= COLLISION_RADIUS:
                _fight(human, other, tick, dt) #are they colliding?fight!

def _fight(human, zombie, tick, dt): 
    human["life_value"] = max(0, human["life_value"] - zombie["strength"] * ATTACK_SCALE * dt) 
    zombie["life_value"] = max(0, zombie["life_value"] - human["strength"] * ATTACK_SCALE * dt)

    if human["life_value"] <= 0:
        human["health_status"] = "dead" #standard death
        print(f"tick {tick}: human #{human['agent_id']} died fighting zombie #{zombie['agent_id']} (health depleted)")
    if zombie["life_value"] <= 0:
        zombie["health_status"] = "dead"
        print(f"tick {tick}: zombie #{zombie['agent_id']} killed by human #{human['agent_id']} (health depleted)")

def check_bites(agents, tick):

    grid = _build_grid(agents) #grid again 
    for human in agents:
        if human["agent_type"] != "human" or human["health_status"] not in ("healthy", "exposed"):
            continue #infected turned or dead dont roll
        for other in _nearby(human, grid): #9 cells out again 
            if other["agent_type"] != "zombie" or other["health_status"] == "dead":
                continue #dead zombies dont roll
            if math.hypot(human["x"] - other["x"], human["y"] - other["y"]) <= COLLISION_RADIUS:
                _attempt_bite(human, other, tick) #roll dice for attempt bite 
                break #humans cant have more than one bite attempt.

def _attempt_bite(human, zombie, tick): 
    if random.uniform(0, 100) < human["susceptibility"]: #if roll succeeds 
        if human["medicine_count"] > 0: #if they have more than one dose
            human["medicine_count"] -= 1 #they use it on themselves but the dose disapears 
            drop = random.uniform(MEDICINE_POTENCY_MIN, MEDICINE_POTENCY_MAX) #they benefit from the dose like any other 
            human["susceptibility"] = max(0, human["susceptibility"] - drop)
            human["life_value"] = min(100, human["life_value"] + MEDICINE_HEAL)
            print(f"tick {tick}: human #{human['agent_id']} self-administers their dose against a bite from "
                  f"zombie #{zombie['agent_id']} -- susceptibility now {human['susceptibility']:.1f}%, "
                  f"life now {human['life_value']:.1f}, 0 doses left")
            return #healed 
        human["health_status"] = "infected"
        human["status_timer"] = 0 #starts turn 
        print(f"tick {tick}: human #{human['agent_id']} infected by zombie #{zombie['agent_id']} "
              f"(susceptibility {human['susceptibility']:.1f}%)")
    else: #roll fails
        human["health_status"] = "exposed" #survived a bite
        human["status_timer"] = 0 #starts a status timer on exposed 
        human["bites_survived"] += 1 #bite survived counter 
        human["susceptibility"] = max(0, human["susceptibility"] - SUSCEPTIBILITY_DECAY) #suceptibility
        print(f"tick {tick}: human #{human['agent_id']} resisted a bite from zombie #{zombie['agent_id']} "
              f"(bite #{human['bites_survived']} survived, susceptibility now {human['susceptibility']:.1f}%)")

#---------------------spatial grid---------------

def _cell(agent): #which grid cell 
    return int(agent["x"] // CELL_SIZE), int(agent["y"] // CELL_SIZE) 

def _build_grid(agents): #groups agents in cell 
    grid = {}
    for agent in agents:
        grid.setdefault(_cell(agent), []).append(agent) #creates and updates 
    return grid

def _in_lab(agent): #if agents in the lab
    x0, y0, x1, y1 = LAB_BOUNDS
    return x0 <= agent["x"] <= x1 and y0 <= agent["y"] <= y1
#---------------------------Priorities0-0------------------
def _human_target(agent, grid, agents): 

    zombie = _nearest_of_type(agent, grid, "zombie") #1
    if zombie:
        return zombie, not agent["fighting"], None #num 1 priority humans take action with nearby zombies 
    if agent["medicine_count"] > 0: #if med count above 1 
        patient = _nearest_infected(agent, agents) #brute force but less agents needed to be checked 
        return (patient, False, None) if patient else (None, False, None) #researchers seek 
    if agent["intelligence"] > INTELLIGENCE_THRESHOLD and agent["health_status"] in ("healthy", "exposed"):
        if _in_lab(agent):
            return None, False, LAB_BOUNDS #once at lab roam within bounds 
        return LAB_POINT, False, None 
    return None, False, None #wander 

def _nearby(agent, grid): 

    acol, arow = _cell(agent) #agents own cell 
    for dcol in (-1, 0, 1): #column 
        for drow in (-1, 0, 1): #own row
            for other in grid.get((acol + dcol, arow + drow), []): #empty list 
                if other is not agent: #skip self
                    yield other

def _nearest_matching(agent, grid, predicate): #closest other agent within PERCEPTION_RADIUS
    nearest, nearest_dist = None, PERCEPTION_RADIUS #starts
    for other in _nearby(agent, grid):
        if not predicate(other):
            continue #skip anyone that doesn't count
        dist = math.hypot(other["x"] - agent["x"], other["y"] - agent["y"])
        if dist <= nearest_dist:
            nearest, nearest_dist = other, dist #new closest match
    return nearest #if no one none 

def _nearest_of_type(agent, grid, wanted_type): #closest living agent of interest
    return _nearest_matching(agent, grid, lambda a: a["agent_type"] == wanted_type and a["health_status"] != "dead")

def _nearest_infected(agent, agents): #closest infected human within medicine radius 

    nearest, nearest_dist = None, MEDICINE_SEARCH_RADIUS
    for other in agents:
        if other is agent or other["agent_type"] != "human" or other["health_status"] != "infected":
            continue
        dist = math.hypot(other["x"] - agent["x"], other["y"] - agent["y"])
        if dist <= nearest_dist:
            nearest, nearest_dist = other, dist
    return nearest

#---------------------movement--------------------

def _move_toward(agent, target, speed): #steps one agent directly toward another agent's current position
    dx, dy = target["x"] - agent["x"], target["y"] - agent["y"] #pointing at the target
    dist = math.hypot(dx, dy)
    if dist > 0: #dont divide by zero (dont break)
        agent["heading"] = math.atan2(dy, dx) #face the direction actually being moved
        agent["x"] += dx / dist * speed 
        agent["y"] += dy / dist * speed

def _move_away(agent, target, speed): #move away from target 
    dx, dy = agent["x"] - target["x"], agent["y"] - target["y"] #flipped 
    dist = math.hypot(dx, dy)
    if dist > 0:
        agent["heading"] = math.atan2(dy, dx)
        agent["x"] += dx / dist * speed
        agent["y"] += dy / dist * speed

def _wander(agent, speed, dt): #no target wander 
    agent["heading"] += random.uniform(-TURN_RATE, TURN_RATE) * dt #random position 
    agent["x"] += math.cos(agent["heading"]) * speed #step forward 
    agent["y"] += math.sin(agent["heading"]) * speed

def _clamp(value, low, high): #keeps it there 
    return max(low, min(high, value))
