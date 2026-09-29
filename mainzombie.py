import json #turn agent/status data into text the browser can read
import threading #the clock runs on its own thread, separate from the web server
import time #for the clock thread's sleep-based ticking
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer #stdlib web server 
import db #Postgres
import simulation #all the movement/infection/combat/research logic

#---------------------database-----------------------------=
db.create_table() #creates the table unless one already exists then it doesnt
db.seed_agents() #wipes and rerolls a totally random population every single launch, no fixed seed
agents = db.load_agents() #pulls the agent rows out of postgres and turns them into python dicts
simulation.init_state(agents) #bolt on the runtime-only fields (heading, timers, etc.) not stored in Postgres

#---------------------clock ----------------
MOVE_INTERVAL = 0.1 #0.1 seconds between movements less jumpy agents
STEPS_PER_TICK = round(1 / MOVE_INTERVAL) #ticks are one second 

tick = 0 #total ticks 
paused = False #post pause feautures 
medicine_produced = 0 #running total of doses ever finished, across the whole lab
history = [] #log of things like ticks and human and zombie count
agents_lock = threading.Lock() #protects against data inconsistency, stops a request from reading an agent mid update

def human_count(): #live human count
    return sum(1 for a in agents if a["agent_type"] == "human" and a["health_status"] != "dead") #dont delete dead guys

def zombie_count(): #zombie count but also dont delete dead guys
    return sum(1 for a in agents if a["agent_type"] == "zombie" and a["health_status"] != "dead")

def medicine_in_circulation(): #total medicine count produced and held 
    return sum(a["medicine_count"] for a in agents if a["agent_type"] == "human")

def run_clock(): #moves agents every move interval and make one tick
    global tick, medicine_produced #declared globally
    steps_this_tick = 0 #steps as miliseonds 
    while True: #indefinite simulation... even if everyone dies 
        time.sleep(MOVE_INTERVAL) #wait between steps (not ticks steps are smaller within ticks as defined in move interval and steps per tick)
        if paused:
            continue #no movement "time" doesnt progress
        with agents_lock: #protects against data misreads 
            simulation.step(agents, dt=MOVE_INTERVAL, night=simulation.is_night(tick)) #night 
            simulation.check_collisions(agents, tick, dt=MOVE_INTERVAL) #life drain 
        steps_this_tick += 1 #how many steps is this tick
        if steps_this_tick < STEPS_PER_TICK: #if steps this tick less than steps per tick main
            continue #0.1 seconds 
        steps_this_tick = 0 #full tick done reset
        tick += 1 #add a tick to counter
        with agents_lock: #once per tick 
            simulation.check_bites(agents, tick) #roll for bite 
            simulation.advance_infections(agents) #3 days (30 ticks) from infected to turned 
            simulation.advance_exposure(agents) #2 ticks from exposed back to healthy if not bitten again
            simulation.deliver_medicine(agents, tick) #cures infection, restores a flat amount of health, and lowers suceptibity by a random ten to 20
            medicine_produced += simulation.advance_research(agents, tick) #researchers make progress
            simulation.decide_stances(agents) #to decide stances roll random against agreession 
        history.append([tick, human_count(), zombie_count()]) #log population indefinitely 

#---------------------html ---------------------------
HOST, PORT = "localhost", 8000 #define port not 8080

class ZombieSimHandler(BaseHTTPRequestHandler): #request for one
    def log_message(self, format, *args): #log then escape 
        pass #escape

    def do_GET(self): 
        if self.path == "/" or self.path == "/index.html": #path to view html
            self._serve_file("index.html", "text/html") #html page <3 
        elif self.path == "/agents": #agents on the plane
            with agents_lock: #snapshot but dont break the sim even if its mid tick
                self._serve_json(agents) #json of agent 
        elif self.path == "/status": #tick and population status/ sim status
            self._serve_json({ #json to display
                "tick": tick, #tick count 
                "paused": paused, #html button pause is python pause 
                "day": simulation.day_number(tick), #what day?
                "night": simulation.is_night(tick), #nightime?
                "human_count": human_count(), #how many people (._.)
                "zombie_count": zombie_count(), #how many zombies 
                "medicine_produced": medicine_produced, #all doses produced 
                "medicine_in_circulation": medicine_in_circulation(), #how many that are still carried 
                "research_ticks": simulation.RESEARCH_TICKS, #ticks needed to finish a dose so frontend can turn progress into a percent
                "lab_bounds": simulation.LAB_BOUNDS, #100x100 box centered on the map, scales itself with width/height
                "day_length": simulation.DAY_LENGTH, #for the chart to depict "time"
                "width": db.WIDTH, #display width of simulation
                "height": db.HEIGHT, #display height of simulation
                "history": history, #full log of the chart across one chart 
            })
        else:
            self.send_response(404) #404 not found :)
            self.end_headers() 

    def do_POST(self):
        global paused #declare globaly 
        if self.path == "/pause": #pause or not pause (action behind button)
            paused = not paused #on/off 
            self._serve_json({"paused": paused}) #paused 
        else:
            self.send_response(404) #404 not found :(
            self.end_headers() 

    def _serve_file(self, filename, content_type): #reads and serves 
        with open(filename, "rb") as f: #R is for reading 
            body = f.read() #read and return bytes
        self.send_response(200) #response header
        self.send_header("Content-Type", content_type) #content type
        self.end_headers() 
        self.wfile.write(body) #actually sends the bytes back to the browser

    def _serve_json(self, data): #json helper
        body = json.dumps(data).encode() #dict list as mentioned above for agents aggression
        self.send_response(200) 
        self.send_header("Content-Type", "application/json") #oh its a json file
        self.end_headers()
        self.wfile.write(body)

#---------------------end and return <3------------------------
if __name__ == "__main__": 
    threading.Thread(target=run_clock, daemon=True).start() #clock runs web runs neither are negotiable 
    server = ThreadingHTTPServer((HOST, PORT), ZombieSimHandler) #ignore slow guys
    print(f"Zombie Sim running at http://{HOST}:{PORT}") #accesibility 
    try:
        server.serve_forever() 
    except KeyboardInterrupt: #end it. 
        server.shutdown()
