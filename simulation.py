import concurrent.futures as cf #call from sleep
import heapq, math, multiprocessing, os, random, sys #heap, math, pools, settings, random, system
from collections import defaultdict #empty li

#--------------------------- data checks and settings
class InvalidState(ValueError): #errors in data
    pass #dw 

# ------------------------- settings from the environment
def env_number(name, default, lo, hi, cast=int): #error catch for common errors
    raw = os.environ.get(name) #text from enviornment 
    if raw is None or raw == "": #not set
        return default #use the default
    try: #try to turn the text into a number
        value = cast(raw) #convert
    except ValueError: #not a number
        raise ValueError(f"{name}={raw!r} must be a {cast.__name__} between {lo} and {hi}") from None #what is allowed?
    if not (lo <= value <= hi) or value != value: #outside the range
        raise ValueError(f"{name}={raw!r} must be a {cast.__name__} between {lo} and {hi}") #what is allowed?
    return value #good value

# ---------------------------------------------------------- agent data checks
HUMAN_STATES = {"healthy", "exposed", "infected", "dead"} #a human who turns becomes a zombie
ZOMBIE_STATES = {"turned", "dead"} #the only states a zombie can be in
FLOATS = {"x": None, "y": None, "heading": None, "movement_speed": (0, 1000), "life_value": (0, 100), #(low, high), None = any finite value
          "strength": (0, 1000), "intelligence": (0, 1000), "susceptibility": (0, 100), "aggression": (0, 100)} #the rest of the number fields and their ranges
INTS = ("agent_id", "status_timer", "bites_survived", "lab_progress", "medicine_count") #whole numbers, never negative
FIELDS = ("agent_type", "health_status", "fighting") + tuple(FLOATS) + INTS #every field an agent must have
EPS = 1e-6 #float noise allowed at the edges

def _number(v): return isinstance(v, (int, float)) and not isinstance(v, bool) #int or float, but not True/False

def check_agents(agents, width, height, limit=20): #returns a list of problems, empty when the data is sound
    bad = [] #problems found so far
    def problem(a, text): #adds one problem to the list
        if len(bad) < limit: bad.append(f"agent {a.get('agent_id', '?')}: {text}") #stop collecting after the limit
    if not isinstance(agents, list): return ["agents must be a list"] #not even a list
    seen = set() #ids seen so far
    for a in agents: #check every agent
        if not isinstance(a, dict): problem({}, "is not a dict"); continue #an agent must be a dict
        missing = [f for f in FIELDS if f not in a] #fields that are not there
        if missing: problem(a, f"missing fields {missing}"); continue #cannot check further without them
        kind, status = a["agent_type"], a["health_status"] #type and status
        if kind not in ("human", "zombie"): problem(a, f"unknown type {kind!r}"); continue #only humans and zombies exist
        if status not in (HUMAN_STATES if kind == "human" else ZOMBIE_STATES): problem(a, f"a {kind} cannot be {status!r}") #the status must fit the type
        if type(a["fighting"]) is not bool: problem(a, "fighting must be True or False") #fighting is yes or no
        for f in INTS: #whole number fields
            v = a[f] #the value
            if type(v) is not int or v < 0: problem(a, f"{f} must be a whole number >= 0, got {v!r}") #must be a whole number and not negative
        for f, rng in FLOATS.items(): #number fields
            v = a[f] #the value
            if not _number(v) or not math.isfinite(v): problem(a, f"{f} is not a finite number: {v!r}"); continue #nan and infinity are caught here
            if rng and not (rng[0] - EPS <= v <= rng[1] + EPS): problem(a, f"{f}={v} outside {rng}") #must be inside its range
        if _number(a["x"]) and _number(a["y"]) and math.isfinite(a["x"]) and math.isfinite(a["y"]): #only when the position is a real number
            if not (-EPS <= a["x"] <= width + EPS and -EPS <= a["y"] <= height + EPS): problem(a, f"outside the map at ({a['x']:.1f}, {a['y']:.1f})") #must be on the map
        if _number(a["life_value"]) and (status == "dead") != (a["life_value"] <= 0): problem(a, f"life {a['life_value']} does not match status {status!r}") #dead means 0 life, alive means more than 0
        if type(a["agent_id"]) is int: #only when the id is a whole number
            if a["agent_id"] in seen: problem(a, "duplicate agent_id") #two agents with one id
            seen.add(a["agent_id"]) #remember this id
    return bad #empty list means the data is sound

def check_state(state, limit=20): #the whole saved state: agents + environment + runtime
    if not isinstance(state, dict): return ["state must be a dict"] #the state must be a dict
    missing = [k for k in ("agents", "environment", "runtime") if k not in state] #top level parts that are missing
    if missing: return [f"state is missing {missing}"] #cannot check further
    env, rt = state["environment"], state["runtime"] #shortcuts
    bad = [] #problems found so far
    for k in ("width", "height"): #map size
        v = env.get(k) if isinstance(env, dict) else None #the value, None if environment is broken
        if not _number(v) or not math.isfinite(v) or v <= 0: bad.append(f"environment {k} must be a positive number, got {v!r}") #must be a real positive number
    if bad: return bad #agents cannot be checked without a map size
    for k in ("tick", "steps", "medicine_produced"): #counters
        v = rt.get(k) if isinstance(rt, dict) else None #the value, None if runtime is broken
        if type(v) is not int or v < 0: bad.append(f"runtime {k} must be a whole number >= 0, got {v!r}") #must be a whole number and not negative
    if not isinstance(rt.get("paused"), bool): bad.append("runtime paused must be True or False") #paused is yes or no
    hist = rt.get("history") #population per tick
    if not isinstance(hist, list) or any(not isinstance(r, (list, tuple)) or len(r) != 3 for r in hist): bad.append("runtime history must be a list of [tick, humans, zombies]") #every row is 3 numbers
    return (bad + check_agents(state["agents"], env["width"], env["height"], limit))[:limit] #plus the agent problems, cut to the limit

def require_valid(state, what="state"): #raises InvalidState with the first few problems
    problems = check_state(state) #check everything
    if problems: raise InvalidState(f"{what} is invalid: " + "; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")) #stop, with the first few problems in the message

# ---------------------------------------------------------- rollback
def snapshot(agents): return [dict(a) for a in agents] #agents hold only plain values, so a shallow copy of each dict is a full copy

def restore(agents, snap): agents[:] = [dict(a) for a in snap] #same list object, so every reference sees the old data again


TOTAL = env_number("ZOMBIE_AGENTS", 1000, 4, 100000) #total agents, 1000 = 750 humans + 250 zombies (stage 1 had 475)
NUM_ZOMBIES = TOTAL // 4 #25% of everyone is a zombie at the start
NUM_HUMANS = TOTAL - NUM_ZOMBIES #the other 75% are people
WIDTH = HEIGHT = round(1000 * (TOTAL / 475) ** 0.5) #Width and height of map 1000x1000 at 475 agents, bigger map with more agents so its not more crowded
HUMAN_SPEED = 4.0 #one tick = 4 points a person can move
ZOMBIE_SPEED = 2.6 #slower than humans same thing tho one tick 


#==================== stateless random draws ====================
#stateless random numbers: the draw depends only on (seed, tick, agent id, salt), never on worker count or order
MASK = (1 << 64) - 1 #keeps numbers inside 64 bits
_K1, _K2, _K3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB #splitmix64 constants

SALT_BITE, SALT_POTENCY, SALT_SELF = 1, 3, 4 #one salt per kind of draw so they never reuse a number

def _mix(x): #splitmix64 finalizer, scrambles 64 bits
    x = (x + _K1) & MASK #add a constant
    x = ((x ^ (x >> 30)) * _K2) & MASK #shift, xor, multiply
    x = ((x ^ (x >> 27)) * _K3) & MASK #again with other numbers
    return x ^ (x >> 31) #last shift and xor

def tick_base(seed, tick): #hash once per tick, each draw then needs one more mix
    return _mix(seed ^ _mix(tick)) #seed and tick mixed into one number

def rand01(base, agent_id, salt=0): #uniform in [0, 1)
    return (_mix(base ^ ((agent_id * _K1) & MASK) ^ salt) >> 11) / (1 << 53) #mix in the agent and the salt, top 53 bits as a fraction

#==================== worker pools ====================

def hdf5_file_open(): #True while any HDF5 file is open in this process
    h5py = sys.modules.get("h5py") #never imports it, only looks if someone else did
    if h5py is None: #nobody imported h5py
        return False #so no file can be open
    try: #ask h5py
        return len(h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)) > 0 #count the open files
    except Exception: #h5py could not answer
        return False #treat as not open

def fork_context(): #the pool context for the workers, or None to run inline
    if os.environ.get("ZOMBIE_START_METHOD", "fork").lower() != "fork": #anything but fork (the default) means one core
        return None #one core
    if hdf5_file_open(): #a forked child inherits the open HDF5 state, which h5py documents as dangerous
        return None #one core
    if "fork" not in multiprocessing.get_all_start_methods(): #Windows
        return None #one core
    return multiprocessing.get_context("fork") #fork: workers start as copies of this process

POOL_TIMEOUT = env_number("ZOMBIE_POOL_TIMEOUT", 120.0, 0.1, 86400.0, float) #seconds one parallel call may take before it is abandoned

def run_parallel(fn, items, workers): #fn over items in forked workers, results in item order, or None when the caller must run inline
    ctx = fork_context() #can workers start?
    if ctx is None: #no
        return None #the caller does the work itself
    try: #try to start them
        ex = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) #the pool of worker processes
    except (OSError, ValueError): #the system refused
        return None #the caller does the work itself
    clean = False #becomes True when everything finished
    try: #run the work
        out = list(ex.map(fn, items, timeout=POOL_TIMEOUT)) #fn on every item, results in the same order as the items
        clean = True #finished normally
        return out #the results
    except (cf.process.BrokenProcessPool, cf.TimeoutError, TimeoutError, OSError): #dead worker, hung worker, cannot fork
        return None #the caller does the work itself
    finally: #always
        ex.shutdown(wait=clean, cancel_futures=True) #never wait on a hung or dead pool

#==================== perception: nearest agent search ====================
PERCEPTION_RADIUS = 120 #how far out can an agent see
SEE2 = PERCEPTION_RADIUS * PERCEPTION_RADIUS #squared, so distances never need a square root
FINE = 15.0 #fine cell size
FINE_RINGS = 2 #rings 0..2 around the agent's fine cell, certifies radius 30
COARSE = PERCEPTION_RADIUS / 2 + 0.5 #60.5: a 5x5 block (rings 0..2) certifies radius > 120

def _ring(r): #the cells that are exactly r cells away
    if r == 0: #ring 0
        return [(0, 0)] #is the cell itself
    out = [(dx, dy) for dx in range(-r, r + 1) for dy in (-r, r)] #top and bottom rows of the ring
    out += [(dx, dy) for dy in range(-r + 1, r) for dx in (-r, r)] #left and right sides without the corners
    return out #the ring

RING = [_ring(r) for r in range(FINE_RINGS + 1)] #rings 0, 1 and 2 worked out once
COARSE_CELLS = [o for ring in RING for o in ring] #the 25 cells of the coarse 5x5 block

class TypeIndex: #one type of agent sorted into two grids
    def __init__(self, xs, ys, members): #members: agent indices, in index order
        self.xs, self.ys = xs, ys #all positions
        fine, coarse = {}, {} #fine grid and coarse grid
        for j in members: #every agent of this type
            x, y = xs[j], ys[j] #its position
            k = (int(x // FINE), int(y // FINE)) #its fine cell
            lst = fine.get(k) #who is already in that cell
            if lst is None: #nobody yet
                fine[k] = [j] #start the list
            else: #somebody is
                lst.append(j) #add to the list
            k = (int(x // COARSE), int(y // COARSE)) #its coarse cell
            lst = coarse.get(k) #who is already there
            if lst is None: #nobody yet
                coarse[k] = [j] #start the list
            else: #somebody is
                lst.append(j) #add to the list
        self.fine, self.coarse = fine, coarse #keep both grids

    def nearest(self, x, y): #index of the nearest member within the radius, or -1
        xs, ys, fine = self.xs, self.ys, self.fine #shortcuts
        fx, fy = int(x // FINE), int(y // FINE) #the fine cell of the one asking
        best_i, best_d2 = -1, SEE2 #nothing found yet, the limit is the seeing distance
        for r in range(FINE_RINGS + 1): #ring by ring outward
            for ox, oy in RING[r]: #every cell in this ring
                lst = fine.get((fx + ox, fy + oy)) #agents in that cell
                if lst: #if any
                    for j in lst: #each one
                        dx, dy = xs[j] - x, ys[j] - y #gap in x and y
                        d2 = dx * dx + dy * dy #squared distance
                        if d2 < best_d2 or (d2 == best_d2 and (best_i < 0 or j < best_i)): #closer, or same distance and lower index
                            best_i, best_d2 = j, d2 #new best
            if best_i >= 0 and best_d2 < (r * FINE) ** 2: #the best is closer than anything not looked at yet
                return best_i #everything not scanned yet is at least r * FINE away
        coarse = self.coarse #fine rings proved nothing: scan the coarse 5x5 block
        cx, cy = int(x // COARSE), int(y // COARSE) #the coarse cell of the one asking
        best_i, best_d2 = -1, SEE2 #start over
        for ox, oy in COARSE_CELLS: #the 25 coarse cells around
            lst = coarse.get((cx + ox, cy + oy)) #agents in that cell
            if lst: #if any
                for j in lst: #each one
                    dx, dy = xs[j] - x, ys[j] - y #gap in x and y
                    d2 = dx * dx + dy * dy #squared distance
                    if d2 < best_d2 or (d2 == best_d2 and (best_i < 0 or j < best_i)): #closer, or same distance and lower index
                        best_i, best_d2 = j, d2 #new best
        return best_i #-1 when nobody is in range

class Perception: #the two indexes agents use to look around
    def __init__(self, agents): #built once per step, read only afterwards (workers see it through fork)
        self.agents = agents #keep the list
        xs = [a["x"] for a in agents] #every x
        ys = [a["y"] for a in agents] #every y
        humans, zombies = [], [] #indices by type
        for i, a in enumerate(agents): #every agent
            if a["health_status"] == "dead": #dead
                continue #dead agents are never seen
            t = a["agent_type"] #its type
            if t == "human": #human
                humans.append(i) #goes in the human list
            elif t == "zombie": #zombie
                zombies.append(i) #goes in the zombie list
        self.humans, self.zombies = TypeIndex(xs, ys, humans), TypeIndex(xs, ys, zombies) #one index per type

    def nearest_human(self, agent): #closest living human within the radius, or None
        j = self.humans.nearest(agent["x"], agent["y"]) #index of the nearest human
        return self.agents[j] if j >= 0 else None #the agent, or None

    def nearest_zombie(self, agent): #closest living zombie within the radius, or None
        j = self.zombies.nearest(agent["x"], agent["y"]) #index of the nearest zombie
        return self.agents[j] if j >= 0 else None #the agent, or None

#==================== contact events: workers report, parent decides ====================


COLLISION_RADIUS = 12 #how close it has to be to touch
ATTACK_SCALE = 0.75 #less damage
R2 = COLLISION_RADIUS * COLLISION_RADIUS #squared touch distance
CONTACT_CELL = COLLISION_RADIUS #cell size = interaction radius
BLOCKS_PER_WORKER = 4 #more blocks = better balance, more scheduling overhead
PAR_MIN_COST = 1_000_000 #candidate checks below which a pool is not worth it (measured, calibrate on your machine)
FIX = 1_000_000 #fixed point: damage resolved to 1e-6 life points
_ECTX = {} #what the workers need, set right before the pool forks

def build_contact_grid(agents): #living humans and zombies by cell, plus flat position/strength lists
    humans, zombies = defaultdict(list), defaultdict(list) #cell -> list of agent indices
    xs, ys, st = [0.0] * len(agents), [0.0] * len(agents), [0.0] * len(agents) #position and strength of everyone as flat lists
    for i, a in enumerate(agents): #every agent
        x, y = a["x"], a["y"] #its position
        xs[i], ys[i], st[i] = x, y, a["strength"] #store it
        if a["health_status"] == "dead": #dead
            continue #dead agents never touch anyone
        key = (int(x // CONTACT_CELL), int(y // CONTACT_CELL)) #its grid cell
        (humans if a["agent_type"] == "human" else zombies)[key].append(i) #into the human grid or the zombie grid
    return humans, zombies, xs, ys, st #what the workers need

def plan_blocks(humans, zombies, workers): #estimate cost per cell, pack cells into blocks (largest first)
    cells, total = [], 0 #cells with their cost, and the total cost
    for (cx, cy), hs in humans.items(): #every cell that has humans
        nz = 0 #zombies near this cell
        for dx in (-1, 0, 1): #the 3x3 cells around
            for dy in (-1, 0, 1): #the 3x3 cells around
                z = zombies.get((cx + dx, cy + dy)) #zombies in that cell
                if z: #if any
                    nz += len(z) #count them
        if nz: #only cells where something can happen
            c = len(hs) * nz #cost = humans x nearby zombies
            cells.append((c, (cx, cy))) #remember the cell
            total += c #add to the total
    if not cells: #no human is near a zombie
        return [], 0 #nothing to do
    nb = max(1, workers * BLOCKS_PER_WORKER) #number of blocks
    cells.sort(key=lambda t: (-t[0], t[1])) #deterministic
    heap = [(0, b) for b in range(nb)] #(load, block) pairs, lightest on top
    blocks = [[] for _ in range(nb)] #empty blocks
    for c, key in cells: #biggest cell first
        load, b = heapq.heappop(heap) #take the lightest block
        blocks[b].append(key) #give it the cell
        heapq.heappush(heap, (load + c, b)) #put it back with its new load
    return [b for b in blocks if b], total #the non-empty blocks and the total cost

def _block_events(block): #map + local reduce, runs in a worker, read only
    humans, zombies = _ECTX["humans"], _ECTX["zombies"] #grids from the parent
    xs, ys, st, bitable, k = _ECTX["xs"], _ECTX["ys"], _ECTX["st"], _ECTX["bitable"], _ECTX["k"] #positions, strength, who can be bitten, damage scale
    dmg, first = {}, {} #damage per agent (fixed point), lowest touching zombie per bitable human
    for cx, cy in block: #every cell in my block
        cand = [] #zombies that could touch
        for dx in (-1, 0, 1): #the 3x3 cells around
            for dy in (-1, 0, 1): #the 3x3 cells around
                z = zombies.get((cx + dx, cy + dy)) #zombies there
                if z: #if any
                    cand.extend(z) #add them
        if not cand: #no zombie near
            continue #next cell
        for h in humans[(cx, cy)]: #every human in the cell
            hx, hy, sh, can_bite = xs[h], ys[h], st[h], bitable[h] #its position, strength, and if it can be bitten
            for z in cand: #every nearby zombie
                ddx, ddy = hx - xs[z], hy - ys[z] #gap in x and y
                if ddx * ddx + ddy * ddy <= R2: #squared distance, no sqrt
                    if k: #scale 0 means only bites are wanted
                        dmg[h] = dmg.get(h, 0) + round(st[z] * k * FIX) #zombie hurts human
                        dmg[z] = dmg.get(z, 0) + round(sh * k * FIX) #human hurts zombie
                    if can_bite and (h not in first or z < first[h]): #first zombie, or a lower index
                        first[h] = z #remember it
    return dmg, first #my report

def process_contacts(agents, dt, workers=1): #the referee: one step of contacts
    humans, zombies, xs, ys, st = build_contact_grid(agents) #sort everyone into cells
    bitable = [a["health_status"] in ("healthy", "exposed") for a in agents] #only healthy and exposed humans can be bitten
    blocks, cost = plan_blocks(humans, zombies, workers) #split the work
    if not blocks: #no contact is possible
        return {}, {} #no damage, no bites
    _ECTX.update(humans=humans, zombies=zombies, xs=xs, ys=ys, st=st, bitable=bitable, k=ATTACK_SCALE * dt) #what the workers read
    parts = None #no reports yet
    if workers > 1 and cost >= PAR_MIN_COST: #enough work for the pool
        parts = run_parallel(_block_events, blocks, workers) #None if forking is unavailable or the pool failed
    if parts is None: #pool not used, or it failed
        parts = [_block_events(b) for b in blocks] #do every block here
    dmg, first = {}, {} #totals
    for d, f in parts: #integer + and min are order independent
        for t, v in d.items(): #every damage entry
            dmg[t] = dmg.get(t, 0) + v #add it up per agent
        for h, z in f.items(): #every bite candidate
            if h not in first or z < first[h]: #first one, or a lower index
                first[h] = z #keep the lowest
    return dmg, first #total damage per agent, biting zombie per human

def apply_damage(agents, dmg): #returns the indices that died this step
    dead = [] #who died
    for i in sorted(dmg): #in index order
        a = agents[i] #the agent
        if a["health_status"] == "dead": #already dead
            continue #skip
        a["life_value"] = max(0.0, a["life_value"] - dmg[i] / FIX) #take off the damage, never below 0
        if a["life_value"] <= 0: #no life left
            a["health_status"] = "dead" #dies
            dead.append(i) #note it
    return dead #the deaths

def bite_rolls(first, agents, seed, tick): #roll drawn only for humans that are actually touching a zombie
    base = tick_base(seed, tick) #the number for this tick
    return [(h, z, rand01(base, agents[h]["agent_id"], SALT_BITE) * 100) for h, z in sorted(first.items())] #one 0-100 roll per touched human, in index order

#==================== medicine delivery ====================


MEDICINE_POTENCY_MIN = 10 #susceptibility drop, low end
MEDICINE_POTENCY_MAX = 20 #susceptibility drop, high end
MEDICINE_HEAL = 25 #health is restored max health 100

def potency(base, agent_id, salt=SALT_POTENCY): #10-20, same for any worker count
    return MEDICINE_POTENCY_MIN + (MEDICINE_POTENCY_MAX - MEDICINE_POTENCY_MIN) * rand01(base, agent_id, salt) #between the low end and the high end

def match_rounds(agents): #returns sorted [(carrier index, patient index)]
    carriers = [i for i, a in enumerate(agents) if a["agent_type"] == "human" and a["medicine_count"] > 0] #humans holding a dose
    if not carriers: #nobody has medicine
        return [] #no deliveries
    cells = defaultdict(list) #infected humans by cell
    for i, a in enumerate(agents): #every agent
        if a["agent_type"] == "human" and a["health_status"] == "infected": #infected human
            cells[(int(a["x"] // CONTACT_CELL), int(a["y"] // CONTACT_CELL))].append(i) #goes in its cell
    if not cells: #nobody is infected
        return [] #no deliveries
    free = list(carriers) #carriers who have not given a dose yet
    taken, out = set(), [] #patients already matched, and the pairs
    while free: #rounds until nobody proposes
        best = {} #patient -> (distance squared, carrier)
        for c in free: #every free carrier
            ca = agents[c] #the carrier
            cx, cy = int(ca["x"] // CONTACT_CELL), int(ca["y"] // CONTACT_CELL) #its cell
            near = None #its nearest patient so far
            for dx in (-1, 0, 1): #the 3x3 cells around
                for dy in (-1, 0, 1): #the 3x3 cells around
                    for p in cells.get((cx + dx, cy + dy), ()): #patients in that cell
                        if p == c or p in taken: #not itself, not already matched
                            continue #skip
                        ddx, ddy = ca["x"] - agents[p]["x"], ca["y"] - agents[p]["y"] #gap in x and y
                        d2 = ddx * ddx + ddy * ddy #squared distance
                        if d2 <= R2 and (near is None or (d2, p) < near): #in reach and closer, ties go to the lower index
                            near = (d2, p) #new nearest
            if near is not None: #found a patient
                d2, p = near #its distance and index
                if p not in best or (d2, c) < best[p]: #no proposer yet, or this one is closer
                    best[p] = (d2, c) #keep this proposer
        if not best: #nobody proposed
            break #done
        won = set() #carriers matched this round
        for p, (d2, c) in sorted(best.items()): #every patient with a proposer, in index order
            taken.add(p) #the patient is taken
            won.add(c) #the carrier is used
            out.append((c, p)) #record the pair
        free = [c for c in free if c not in won] #the rest try again
    return sorted(out) #sorted, so the order never depends on the rounds

def _deliver_doses(agents, seed, tick, log=None): #apply the cures
    base = tick_base(seed, tick) #the number for this tick
    for c, p in match_rounds(agents): #every matched pair
        carrier, patient = agents[c], agents[p] #the two agents
        patient["health_status"] = "healthy" #cure
        patient["status_timer"] = 0 #infection timer reset
        drop = potency(base, patient["agent_id"]) #how strong this dose is
        patient["susceptibility"] = max(0, patient["susceptibility"] - drop) #harder to infect, never below 0
        patient["life_value"] = min(100, patient["life_value"] + MEDICINE_HEAL) #heal, never above 100
        carrier["medicine_count"] -= 1 #dose used up
        if log is not None: #a log was asked for
            log.append(f"tick {tick}: human #{carrier['agent_id']} cures human #{patient['agent_id']} "
                       f"-- susceptibility now {patient['susceptibility']:.1f}%, " #rest of the message
                       f"life now {patient['life_value']:.1f}, {carrier['medicine_count']} doses left") #rest of the message

#==================== simulation rules ====================
#integrated midterm
WORKERS = 1 #that many worker processes
SEED = env_number("ZOMBIE_SEED", 42, 0, 2**63 - 1) #seed for the stateless draws (bites, medicine potency)
CHUNK = 2000 #how many agents each worker takes at a time
_CTX = {} #what the workers need to see right before they start

def _work(span): #runs inside a worker
    return _CTX["func"](*span) #call the function on this stack

MIN_CHUNK = 250 #never cut the agents into pieces smaller than this
PAR_MIN_AGENTS = 2500 #below this a pool costs more than it saves (measured, calibrate on your machine)

def pmap(func, n): #cuts the agents into chunks, workers run func on them at the same time, results come back in order
    chunk = CHUNK if WORKERS <= 1 else min(CHUNK, max(MIN_CHUNK, -(-n // (WORKERS * 4)))) #about 4 pieces per worker so the pool can balance
    spans = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)] #chunk start and end
    if WORKERS > 1 and n >= PAR_MIN_AGENTS: #small populations run inline, a pool would cost more than the work
        _CTX["func"] = func #workers fork after this so they already have the current agents
        parts = run_parallel(_work, spans, WORKERS) #None if forking is unavailable or the pool failed
        if parts is not None: #the pool worked
            return [r for part in parts for r in part] #ordered, so the result never depends on the worker count
    return [r for span in spans for r in func(*span)] #one core: same stacks, done here in order

TURN_RATE = 0.6 #the speed on which they turn 
            

                      

DAY_LENGTH = 10 #one day 
NIGHT_START = DAY_LENGTH // 2 #5 ticks day 5 ticks night
TURN_TICKS = 3 * DAY_LENGTH #takes 3 days to turn 
SUSCEPTIBILITY_DECAY = 5 #decay by 5 points
EXPOSED_TICKS = 2 #A person is exposed for 2 ticks 

#--------------Lab Stuff--------------------------------------

LAB_SIZE = 100 #
LAB_BOUNDS = (WIDTH / 2 - LAB_SIZE / 2, HEIGHT / 2 - LAB_SIZE / 2, #lab corner: left, top
              WIDTH / 2 + LAB_SIZE / 2, HEIGHT / 2 + LAB_SIZE / 2) #lab corner: right, bottom
LAB_POINT = {"x": WIDTH / 2, "y": HEIGHT / 2} #lab centre, where researchers walk to
INTELLIGENCE_THRESHOLD = 5 #have to have at least a 5 for research progress
RESEARCH_TICKS = 20 #ticks needed for progress to produce medicine
MEDICINE_SEARCH_RADIUS = 400 #because only a few infected exist on the map the carriers can search more area 

def day_number(tick): #day increments  count 
    return tick // DAY_LENGTH + 1 #add one to the counter

def is_night(tick): #night is the second half of the day
    return tick % DAY_LENGTH >= NIGHT_START #start night during later of the half

def init_state(agents): #fields every agent gets at the start
    for agent in agents: #every agent
        agent["heading"] = random.uniform(0, 2 * math.pi) #direction
        agent["status_timer"] = 0 #status timer vital for temp status like infected of exposed 
        agent["fighting"] = False #fight or flight
        agent["bites_survived"] = 0 #counts of failed bites 
        agent["lab_progress"] = 0 #tick progress made by labs
        agent["medicine_count"] = 0 #doses currently being carried 
        
#_-----------------------fight or flight-------------------

def _sees_zombie(lo, hi): #parallel, does each human have a zombie nearby
    agents, index = _CTX["agents"], _CTX["index"] #what the worker needs
    return [a["agent_type"] == "human" and a["health_status"] != "dead" and index.nearest_zombie(a) is not None #Agent of interest nearby? No?
            for a in agents[lo:hi]] #only this stack of agents

def decide_stances(agents): #fight or flee for every human
    _CTX.update(agents=agents, index=Perception(agents)) #one index per agent type
    sees = pmap(_sees_zombie, len(agents)) #nearest zombie lookups happen in the workers
    for agent, target in zip(agents, sees): #every agent with what it saw
        if agent["agent_type"] != "human" or agent["health_status"] == "dead": #not a living human
            continue #dead people dont do anything
        agent["fighting"] = target and random.uniform(0, 100) < agent["aggression"] #no target nothing happens if target roll random compare against agression

#---------------------------Infection progress-----------------

def advance_infections(agents): #infected humans count toward turning

    for agent in agents: #every agent
        if agent["health_status"] != "infected": #when someone is infected:
            continue #skip
        agent["status_timer"] += 1 #timer counts down (up) to turned 
        if agent["status_timer"] >= TURN_TICKS: #time is up
            agent["agent_type"] = "zombie" #flips agent type 
            agent["health_status"] = "turned" #now turned 
            agent["movement_speed"] = ZOMBIE_SPEED #speed decreases from human to zombie
            agent["status_timer"] = 0 #infection effect expires now turned 

#-------------------_Exposed-------------------------
def advance_exposure(agents): #exposed humans recover
    for agent in agents: #every agent
        if agent["health_status"] != "exposed": #for exposed humans 
            continue #skip
        agent["status_timer"] += 1 #status effect expriation goes down (up)
        if agent["status_timer"] >= EXPOSED_TICKS: #time is up
            agent["health_status"] = "healthy" #back to healthy
            agent["status_timer"] = 0 #reset 

#--------------------research progress ---------

def advance_research(agents, tick): #lab work
    produced = 0 #doses produced 
    for agent in agents: #every agent
        if (agent["agent_type"] != "human" or agent["health_status"] not in ("healthy", "exposed") #must be a living human who is not infected
                or agent["medicine_count"] > 0 #if researcher carying a dose research will not progress 
                or agent["intelligence"] <= INTELLIGENCE_THRESHOLD or not _in_lab(agent)): #inteligence at 5 or above 
            continue #skip
        agent["lab_progress"] += 1 #dose progress 
        if agent["lab_progress"] >= RESEARCH_TICKS: #enough work for a dose
            agent["medicine_count"] += 1 #dose is done added to "inventory"
            agent["lab_progress"] = 0 #reset 
            produced += 1 #add to counter 
            print(f"tick {tick}: human #{agent['agent_id']} finishes a dose of medicine")
    return produced #doses made this tick

def deliver_medicine(agents, tick): #proposal rounds + deterministic matching, see the medicine section above
    log = [] #messages
    _deliver_doses(agents, SEED, tick, log) #match carriers to patients and cure
    for line in log: #every message
        print(line) #show it

def _decide(lo, hi): #parallel, what does each agent see and where is it going
    agents, index, night, infected = _CTX["agents"], _CTX["index"], _CTX["night"], _CTX["infected"] #what the worker needs
    out = [] #one decision per agent
    for agent in agents[lo:hi]: #this stack
        if agent["health_status"] == "dead" or (night and agent["agent_type"] == "human"): #dead, or asleep
            out.append(None) #dead agents stay put / asleep
            continue #next agent
        if agent["agent_type"] == "zombie": #zombie
            target, fleeing, bounds = index.nearest_human(agent), False, None #zombies always chase
        else: #human
            target, fleeing, bounds = _human_target(agent, index, infected) #humans have the full priority list
        out.append(((target["x"], target["y"]) if target else None, fleeing, bounds)) #only send back the x and y of the target
    return out #decisions in agent order

def step(agents, dt=1.0, night=False): #one movement step
    index = Perception(agents) #new call
    infected = [a for a in agents if a["agent_type"] == "human" and a["health_status"] == "infected"] #only infected humans, so carriers dont check every agent
    _CTX.update(agents=agents, index=index, night=night, infected=infected) #workers get these
    decisions = pmap(_decide, len(agents)) #workers find targets
    for agent, decision in zip(agents, decisions): #moving still happens here one at a time
        if decision is None: #nothing to do
            continue #dead agents stay put / asleep
        xy, fleeing, bounds = decision #target spot, run or chase, bounds
        target = {"x": xy[0], "y": xy[1]} if xy else None #rebuild a target the move functions can use
        step_speed = agent["movement_speed"] * dt #distance this step
        if target: #has a target
            _move_away(agent, target, step_speed) if fleeing else _move_toward(agent, target, step_speed) #run from it or go to it
        else: #no target
            _wander(agent, step_speed, dt) #wander 
        x0, y0, x1, y1 = bounds if bounds else (0, 0, WIDTH, HEIGHT) #bounds 
        agent["x"] = _clamp(agent["x"], x0, x1) #agents stuck within graph
        agent["y"] = _clamp(agent["y"], y0, y1) #stay on the map, y

def check_collisions(agents, tick, dt=1.0): #fights: workers find contacts and sum damage per block, parent applies totals
    damage, _ = process_contacts(agents, dt, WORKERS) #fights are simultaneous now, a zombie killed this step still lands its hit
    for i in apply_damage(agents, damage): #everyone whose life hit 0
        print(f"tick {tick}: {agents[i]['agent_type']} #{agents[i]['agent_id']} died (health depleted)")

def check_bites(agents, tick): #one bite attempt per human touching a zombie (lowest zombie index), roll is stateless
    _, first = process_contacts(agents, 0.0, WORKERS) #dt 0: only the bite candidates are needed
    for h, z, roll in bite_rolls(first, agents, SEED, tick): #every touched human with its roll
        _attempt_bite(agents[h], agents[z], tick, roll) #try the bite

def _attempt_bite(human, zombie, tick, roll): #one bite: infect, self cure, or resist
    if roll < human["susceptibility"]: #if roll succeeds (roll is a stateless draw keyed by seed, tick and agent)
        if human["medicine_count"] > 0: #if they have more than one dose
            human["medicine_count"] -= 1 #they use it on themselves but the dose disapears 
            drop = potency(tick_base(SEED, tick), human["agent_id"], SALT_SELF) #they benefit from the dose like any other
            human["susceptibility"] = max(0, human["susceptibility"] - drop) #harder to infect
            human["life_value"] = min(100, human["life_value"] + MEDICINE_HEAL) #heal
            print(f"tick {tick}: human #{human['agent_id']} self-administers their dose against a bite from "
                  f"zombie #{zombie['agent_id']} -- susceptibility now {human['susceptibility']:.1f}%, "
                  f"life now {human['life_value']:.1f}, 0 doses left") #rest of the message
            return #healed 
        human["health_status"] = "infected" #now infected
        human["status_timer"] = 0 #starts turn 
        print(f"tick {tick}: human #{human['agent_id']} infected by zombie #{zombie['agent_id']} "
              f"(susceptibility {human['susceptibility']:.1f}%)") #rest of the message
    else: #roll fails
        human["health_status"] = "exposed" #survived a bite
        human["status_timer"] = 0 #starts a status timer on exposed 
        human["bites_survived"] += 1 #bite survived counter 
        human["susceptibility"] = max(0, human["susceptibility"] - SUSCEPTIBILITY_DECAY) #suceptibility
        print(f"tick {tick}: human #{human['agent_id']} resisted a bite from zombie #{zombie['agent_id']} "
              f"(bite #{human['bites_survived']} survived, susceptibility now {human['susceptibility']:.1f}%)")

#---------------------lab check---------------

def _in_lab(agent): #if agents in the lab
    x0, y0, x1, y1 = LAB_BOUNDS #lab edges
    return x0 <= agent["x"] <= x1 and y0 <= agent["y"] <= y1 #inside the square
#---------------------------Priorities0-0------------------
def _human_target(agent, index, agents): #what a human goes for, in order

    zombie = index.nearest_zombie(agent) #1
    if zombie: #a zombie is in sight
        return zombie, not agent["fighting"], None #num 1 priority humans take action with nearby zombies 
    if agent["medicine_count"] > 0: #if med count above 1 
        patient = _nearest_infected(agent, agents) #brute force but less agents needed to be checked 
        return (patient, False, None) if patient else (None, False, None) #researchers seek 
    if agent["intelligence"] > INTELLIGENCE_THRESHOLD and agent["health_status"] in ("healthy", "exposed"): #smart enough and well enough
        if _in_lab(agent): #already inside
            return None, False, LAB_BOUNDS #once at lab roam within bounds 
        return LAB_POINT, False, None #walk to the lab
    return None, False, None #wander 

def _nearest_infected(agent, agents): #closest infected human within medicine radius 

    nearest, nearest_dist = None, MEDICINE_SEARCH_RADIUS #nothing yet, the limit is the search radius
    for other in agents: #everyone
        if other is agent or other["agent_type"] != "human" or other["health_status"] != "infected": #not itself, only infected humans
            continue #skip
        dist = math.hypot(other["x"] - agent["x"], other["y"] - agent["y"]) #distance
        if dist <= nearest_dist: #closer
            nearest, nearest_dist = other, dist #new nearest
    return nearest #the patient, or None

#---------------------movement--------------------

def _move_toward(agent, target, speed): #steps one agent directly toward another agent's current position
    dx, dy = target["x"] - agent["x"], target["y"] - agent["y"] #pointing at the target
    dist = math.hypot(dx, dy) #distance
    if dist > 0: #dont divide by zero (dont break)
        agent["heading"] = math.atan2(dy, dx) #face the direction actually being moved
        agent["x"] += dx / dist * speed #move in x
        agent["y"] += dy / dist * speed #move in y

def _move_away(agent, target, speed): #move away from target 
    dx, dy = agent["x"] - target["x"], agent["y"] - target["y"] #flipped 
    dist = math.hypot(dx, dy) #distance
    if dist > 0: #dont divide by zero
        agent["heading"] = math.atan2(dy, dx) #face away
        agent["x"] += dx / dist * speed #move in x
        agent["y"] += dy / dist * speed #move in y

def _wander(agent, speed, dt): #no target wander 
    agent["heading"] += random.uniform(-TURN_RATE, TURN_RATE) * dt #random position 
    agent["x"] += math.cos(agent["heading"]) * speed #step forward 
    agent["y"] += math.sin(agent["heading"]) * speed #step forward in y

def _clamp(value, low, high): #keeps it there 
    return max(low, min(high, value)) #low <= value <= high

def bench(): #short headless runs at a few sizes, shown at the bottom of the html
    import json, os, subprocess, sys #tools only this function needs
    rows = [] #table rows
    for n in (1000, 5000, 10000, 20000): #agent counts, small to non-trivial
        for w in (1, os.cpu_count()): #1 worker vs all of them
            out = subprocess.run([sys.executable, "benchmark.py", "--agents", str(n), "--workers", str(w), "--ticks", "1"], capture_output=True, text=True).stdout #one tick in a separate process
            rows.append([n, w, json.loads(out.strip().splitlines()[-1])["s_per_tick"]]) #agents, workers, seconds per tick
    return rows #for the page

if __name__ == "__main__": #python3 simulation.py starts the whole thing and opens the page
    import zombie_storage #the server lives there
    zombie_storage.main() #start everything
