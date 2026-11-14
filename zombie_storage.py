#imports
import json, os, random, sys, threading, time, traceback, webbrowser #standard tools
from abc import ABC, abstractmethod #for the store interface
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer #web server
import simulation #the rules

simulation.WORKERS = simulation.env_number("ZOMBIE_WORKERS", 1, 1, 256) #more than 1 = parallel neighbour finding and events from simulation.py
PORT, STEP, STEPS_PER_TICK = 8001, 0.1, 10 #define port for web step and steps for ticks (crucial for function)

# store
class StoreCorrupt(RuntimeError): #the saved file can't be read
    pass #nothing extra, just a named error

class StateStore(ABC): #what every store must be able to do
    
    @abstractmethod #program fails if theres no load
    def load(self): """return the whole state dict, or None if nothing is stored""" #read
    @abstractmethod #program fails if theres no save.
    def save(self, state): """persist the whole state dict""" #write

class InMemoryStore(StateStore): #keeps the state in memory only
    def __init__(self): self.state = None #nothing is stored yet
    def load(self): return self.state #loads real state
    def save(self, state): self.state = state #updates real state

class HDF5Store(StateStore): #keeps the state in one HDF5 file
    def __init__(self, path="zombie_state.h5"): #the file name
        import h5py, numpy #only needed here, so --memory runs without them installed
        self.h5, self.np, self.path = h5py, numpy, path #keep the libraries and the path

    @staticmethod #no self needed
    def _text(x): return x.decode("utf-8") if isinstance(x, bytes) else str(x) #h5py hands back bytes or str depending on the type

    def _column(self, values): #one list of python values -> one typed numpy column
        np = self.np #shortcut
        if all(type(v) is bool for v in values): return np.array(values, dtype=bool) #all True/False
        if all(isinstance(v, str) for v in values): #fixed width utf-8, compresses and reads faster than variable length strings
            enc = [v.encode("utf-8") for v in values] #text to bytes
            return np.array(enc, dtype=f"S{max(1, max(map(len, enc), default=1))}") #fixed width text column
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values): #mixed int/float columns become float64
            return np.array(values, dtype=np.int64 if all(type(v) is int for v in values) else np.float64) #whole numbers, or decimals
        raise TypeError("a column must hold only bool, only str, or only numbers") #mixed types are refused

    def _put_agents(self, group, agents): #one column per agent field
        fields = list(agents[0]) if agents else [] #field names from the first agent
        for a in agents: #every agent
            if a.keys() != set(fields): raise ValueError("every agent must have the same fields") #must have the same fields
        if fields: group.attrs.create("fields", data=fields, dtype=self.h5.string_dtype()) #keeps the field order
        for k in fields: group.create_dataset(k, data=self._column([a[k] for a in agents])) #write each column

    def _put(self, group, name, v): #dicts become groups, numbers/strings/number lists become datasets
        np = self.np #shortcut
        if "/" in name: raise ValueError(f"'/' is not allowed in a key: {name!r}") #/ means a folder in HDF5
        if isinstance(v, dict): #a dict
            g = group.create_group(name); g.attrs["kind"] = "dict" #becomes a group
            for k, x in v.items(): self._put(g, str(k), x) #with each entry inside
        elif isinstance(v, (list, tuple)): #a list or tuple
            arr = np.array(v) if len(v) else np.empty((0,), dtype=np.float64) #to an array, an empty list becomes an empty array
            if arr.dtype.kind not in "biuf": raise TypeError(f"{name!r}: lists must hold only numbers") #numbers only
            group.create_dataset(name, data=arr).attrs["kind"] = "tuple" if isinstance(v, tuple) else "list" #one dataset that remembers list or tuple
        elif isinstance(v, bool): group.create_dataset(name, data=v).attrs["kind"] = "bool" #True/False
        elif isinstance(v, int): group.create_dataset(name, data=np.int64(v)).attrs["kind"] = "int" #whole number
        elif isinstance(v, float): group.create_dataset(name, data=np.float64(v)).attrs["kind"] = "float" #decimal
        elif isinstance(v, str): group.create_dataset(name, data=v, dtype=self.h5.string_dtype()).attrs["kind"] = "str" #text
        else: raise TypeError(f"{name!r}: cannot store a {type(v).__name__}") #anything else is refused

    def _get_agents(self, g): #columns back into agent dicts
        fields = [self._text(x) for x in g.attrs.get("fields", [])] #field names in their order
        cols = [] #one list per field
        for k in fields: #every field
            arr = g[k][()] #the whole column
            cols.append([b.decode("utf-8") for b in arr.tolist()] if arr.dtype.kind == "S" else arr.tolist()) #tolist gives plain python types
        return [dict(zip(fields, row)) for row in zip(*cols)] #one dict per row

    def _get(self, obj): #reads groups and values back
        if isinstance(obj, self.h5.Group): return {k: self._get(obj[k]) for k in obj} #group -> dict
        kind, raw = self._text(obj.attrs["kind"]), obj[()] #what it was, and the value
        if kind in ("list", "tuple"): #list or tuple
            lst = raw.tolist() #array to list
            return tuple(lst) if kind == "tuple" else lst #tuple if it was one
        if kind == "str": return self._text(raw) #text
        return {"bool": bool, "int": int, "float": float}[kind](raw) #back to bool, int or float

    def _read(self, path): #the whole file to a state dict
        with self.h5.File(path, "r") as f: #closed again before this returns
            state = {k: self._get_agents(f[k]) if k == "agents" else self._get(f[k]) for k in f} #agents by column, the rest value by value
        return state or None #None if the file is empty

    def load(self): #read the saved state
        if not os.path.exists(self.path): return None #nothing stored yet
        try: return self._read(self.path) #read it
        except Exception as e: #h5py raises OSError for empty, truncated or garbage files, KeyError/ValueError for damaged contents
            raise StoreCorrupt(f"cannot read the saved state in {self.path}: {type(e).__name__}: {e}") #a clear error

    def save(self, state): #write the state
        tmp = self.path + ".tmp" #written here first
        try: #try to write
            with self.h5.File(tmp, "w") as f: #closed again before the rename
                for k, v in state.items(): #each part of the state
                    if k == "agents": self._put_agents(f.create_group("agents"), v) #agents by column
                    else: self._put(f, k, v) #the rest
            os.replace(tmp, self.path) #atomic: readers see the old file or the new one, never half of one
        except BaseException: #anything went wrong
            if os.path.exists(tmp): os.remove(tmp) #remove the half written temp file
            raise #pass the error on

# simulation
class Simulation: #runs the simulation and keeps its state
    def __init__(self, store, resume=False): #empty? if so make new one
        self.store, self.lock = store, threading.Lock() #the store, and a lock so two threads never change the state at once
        self.last_error = None #set when a step failed and was rolled back
        loaded = store.load() if resume else None #raises StoreCorrupt when the file cannot be read
        if loaded: #resumed
            simulation.require_valid(loaded, "the saved state") #never continue from malformed data, and never silently replace it
            self.state = loaded #the one read of the whole file
        else: #fresh start
            self.state = self.new_state() #new population
            self._save() #builds over the current state 

    def _save(self): #whole state to the store, a failure is reported and the run keeps going
        try: self.store.save(self.state) #write
        except Exception as e: #it failed
            self.last_error = f"could not save: {type(e).__name__}: {e}" #remember why
            traceback.print_exc() #show where

    def save_now(self): #used when the program stops
        with self.lock: self._save() #save under the lock

    @staticmethod #no self needed
    def new_state(): #the starting population
        agents = [] #hold agents 
        for kind, n, status, speed in (("human", simulation.NUM_HUMANS, "healthy", simulation.HUMAN_SPEED), #human fundemental 
                                       ("zombie", simulation.NUM_ZOMBIES, "turned", simulation.ZOMBIE_SPEED)): #zombie fundemental
            for _ in range(n): #that many
                agents.append({"agent_id": len(agents) + 1, "agent_type": kind, "health_status": status, #generate basic agent stats 
                               "x": random.uniform(0, simulation.WIDTH), "y": random.uniform(0, simulation.HEIGHT), #1000x1000 grid 
                               "movement_speed": speed, "life_value": 100, #movement speed and total life value 
                               "strength": random.uniform(5, 15), "intelligence": random.uniform(1, 10), #strength determines fighting agents above 5 can research 
                               "susceptibility": random.uniform(80, 99) if kind == "human" else 0, #succeptibilty comparison of bite attempt
                               "aggression": random.uniform(0, 100)}) #roles then closes flight or flight
        simulation.init_state(agents) #add runtime 
        return {"agents": agents, #return value
                "environment": {"width": simulation.WIDTH, "height": simulation.HEIGHT, "lab_bounds": simulation.LAB_BOUNDS, #width height lab bounds
                                "day_length": simulation.DAY_LENGTH, "research_ticks": simulation.RESEARCH_TICKS}, #day length research ticks
                "runtime": {"tick": 0, "steps": 0, "paused": True, "medicine_produced": 0, "history": []}} #tick steps paused medicine produced

    def advance(self): #one step
        with self.lock: #one thread at a time
            s = self.state #in memory, nothing is read from disk here
            a, rt = s["agents"], s["runtime"] #shortcuts
            if rt["paused"]: return #paused
            snap, rt_snap, hist_len = simulation.snapshot(a), dict(rt), len(rt["history"]) #a step is all or nothing
            ticked = False #becomes True when a tick finishes
            try: #run the step
                simulation.step(a, dt=STEP, night=simulation.is_night(rt["tick"])) #night
                simulation.check_collisions(a, rt["tick"], dt=STEP) #drain health and bite attempts
                rt["steps"] += 1 #step up
                if rt["steps"] >= STEPS_PER_TICK: #10 steps per tick
                    rt["steps"], rt["tick"] = 0, rt["tick"] + 1 #passage of time
                    simulation.check_bites(a, rt["tick"]) #bite roll
                    simulation.advance_infections(a) #status count down
                    simulation.advance_exposure(a) #status count down
                    simulation.deliver_medicine(a, rt["tick"]) #deliver medecine 
                    rt["medicine_produced"] += simulation.advance_research(a, rt["tick"]) #doses finished 
                    simulation.decide_stances(a) #fight or flight 
                    rt["history"].append([rt["tick"], self.count(a, "human"), self.count(a, "zombie")]) #tick living humans
                    ticked = True #a tick finished
                simulation.require_valid(s, "the state after this step") #nan, out of the map, wrong status... caught before it is saved
            except Exception as e: #roll back to before the step, pause, and keep the server alive
                simulation.restore(a, snap) #agents back
                del rt["history"][hist_len:] #history back
                rt.clear(); rt.update(rt_snap); rt["paused"] = True #counters back, and pause
                self.last_error = f"{type(e).__name__}: {e}" #remember why
                traceback.print_exc() #show where
                self._save() #the rolled back, paused state
            if ticked: self._save() #once per tick, not once per step

    @staticmethod #no self needed
    def count(agents, kind): #living agents of one kind
        return sum(1 for a in agents if a["agent_type"] == kind and a["health_status"] != "dead") #agents who arent dead 

    def toggle_pause(self): #pause or resume
        with self.lock: #with lock 
            rt = self.state["runtime"] #runtime
            rt["paused"] = not rt["paused"] #switch
            if not rt["paused"]: self.last_error = None #resuming clears an old failure message
            self._save() #stores change
            return rt["paused"] #sends back new value

    def agents(self): #for the page
        with self.lock: return [dict(a) for a in self.state["agents"]] #a copy, so the clock can keep moving agents while this one is sent

    def status(self): #for the page
        with self.lock: #under the lock
            s = self.state #shortcut
            a, rt, env = s["agents"], s["runtime"], s["environment"] #agents, runtime, enviornment nickname 
            return {"tick": rt["tick"], "paused": rt["paused"], "day": simulation.day_number(rt["tick"]), #day number 
                    "night": simulation.is_night(rt["tick"]), "human_count": self.count(a, "human"), #nightime darken, human count 
                    "zombie_count": self.count(a, "zombie"), "medicine_produced": rt["medicine_produced"], #living zombies, med produced
                    "medicine_in_circulation": sum(x["medicine_count"] for x in a if x["agent_type"] == "human"), #total carried doses
                    "research_ticks": env["research_ticks"], "lab_bounds": env["lab_bounds"], #research progress to percent 
                    "day_length": env["day_length"], "width": env["width"], "height": env["height"], #ticket per day day length grid
                    "history": list(rt["history"]), "error": self.last_error} #closes, error is None unless a step failed and was rolled back

# web
class Handler(BaseHTTPRequestHandler): #answers the browser
    log_message = lambda *a: None #collects arguments none is a return
    def reply(self, body, ctype="application/json", status=200): #default json 
        self.send_response(status); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)   #send status 200 write body bytes
    def do_GET(self): #the page asks for something
        try: self._get() #answer it
        except Exception as e: #a bad request or a broken endpoint must not kill the connection thread silently
            traceback.print_exc() #show where
            self.reply(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), status=500) #tell the page
    def _get(self): #which answer
        if self.path in ("/", "/index.html"): self.reply(open("index.html", "rb").read(), "text/html") #if path in / and/index read raw bytes and sends html
        elif self.path == "/agents": self.reply(json.dumps(sim.agents(), allow_nan=False).encode()) #for agents convert agent list to JSON encode to bytes sends to sim 
        elif self.path == "/status": self.reply(json.dumps(sim.status(), allow_nan=False).encode()) #same thing but for status 
        elif self.path == "/bench": self.reply(json.dumps(simulation.bench()).encode()) #runs the short tests
        else: self.send_error(404) #404 not found 
    def do_POST(self): #the page sends something
        try: self._post() #handle it
        except Exception as e: #it failed
            traceback.print_exc() #show where
            self.reply(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), status=500) #tell the page
    def _post(self): #only pause exists
        if self.path != "/pause": return self.send_error(404) #404 not found rejects post 
        self.reply(json.dumps({"paused": sim.toggle_pause()}).encode()) #toggle pause button

def main(): #starts everything: store, clock, web server, and opens the page
    global sim #the web handler above reads this
    os.chdir(os.path.dirname(os.path.abspath(__file__))) #index.html and benchmark.py are found wherever this was started from
    if "--memory" in sys.argv: store = InMemoryStore() #nothing on disk, gone when the program stops
    else: #default: one HDF5 file, --hdf5 other.h5 picks a different one
        i = sys.argv.index("--hdf5") + 1 if "--hdf5" in sys.argv else 0 #where the file name sits after --hdf5, 0 if not given
        store = HDF5Store(sys.argv[i] if 0 < i < len(sys.argv) and not sys.argv[i].startswith("--") else "zombie_state.h5") #that file, or the default name
    sim = Simulation(store, resume="--resume" in sys.argv) #is --resume present within sys.argv?
    def clock(): #runs in the background
        while True: #forever
            time.sleep(STEP) #wait one step
            try: sim.advance() #time advances while sleeping 
            except Exception as e: #advance() handles step failures itself, this covers the store failing to load
                sim.last_error = f"{type(e).__name__}: {e}"; traceback.print_exc(); time.sleep(1) #never let the clock thread die silently
    threading.Thread(target=clock, daemon=True).start() #runs slower sleep then work
    url = f"http://localhost:{PORT}" #the address
    print(f"Zombie sim running at {url}") #print the link
    if "--no-browser" not in sys.argv: threading.Timer(0.5, webbrowser.open, [url]).start() #open the page by itself once the server is up
    try: ThreadingHTTPServer(("localhost", PORT), Handler).serve_forever() #Server with no expiration date hosted locally
    except KeyboardInterrupt: pass #manuel cancel
    finally: #on the way out
        sim.save_now() #the ticks since the last save

if __name__ == "__main__": main() #start
