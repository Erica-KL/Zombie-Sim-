import random #starting stats
import psycopg2 #postgres

WIDTH, HEIGHT = 1000, 1000 #Width and height of map 1000x1000
NUM_HUMANS = 350 #how many people 75% of 500 
NUM_ZOMBIES = 125 #how many zombies 25% of 500
HUMAN_SPEED = 4.0 #one tick = 4 points a person can move
ZOMBIE_SPEED = 2.6 #slower than humans same thing tho one tick 

def get_connection(): #new connection 
    return psycopg2.connect(dbname="zombie_sim") #create local "temperary" db

def create_table(): 
    conn = get_connection() #calls def get_connection
    cur = conn.cursor() #sql commands 
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id       SERIAL PRIMARY KEY,        -- auto-incrementing unique id per agent
            agent_type     VARCHAR(10) NOT NULL,      -- "human" or "zombie"
            health_status  VARCHAR(10) NOT NULL,      -- healthy/exposed/infected/turned/dead
            x              FLOAT NOT NULL,            -- spawn position, px
            y              FLOAT NOT NULL,            -- spawn position, px
            movement_speed FLOAT NOT NULL,            -- px moved per full tick
            life_value     FLOAT NOT NULL,             -- hit points, drained by fighting
            strength       FLOAT NOT NULL,             -- damage dealt per tick in a fight
            intelligence   FLOAT NOT NULL,             -- gates who's smart enough to research
            susceptibility FLOAT NOT NULL DEFAULT 0,   -- % chance a bite actually infects
            aggression     FLOAT NOT NULL DEFAULT 50   -- % chance a human fights instead of flees
        );
    """) 
    cur.execute("ALTER TABLE agents ADD COLUMN IF NOT EXISTS aggression FLOAT NOT NULL DEFAULT 50;") #sql command 
    
    conn.commit() #writes changes 
    cur.close() #close cursor
    conn.close() #close connection 

def seed_agents():
    conn = get_connection() #new connection 
    cur = conn.cursor() #connect cursor again 
    cur.execute("TRUNCATE agents RESTART IDENTITY;") #sql set peeps to one 
    for _ in range(NUM_HUMANS): #every human gets their own row. 
        _insert_agent(cur, "human", "healthy", speed=HUMAN_SPEED) #people's status and speed (4)
    for _ in range(NUM_ZOMBIES): #zombies get a row
        _insert_agent(cur, "zombie", "turned", speed=ZOMBIE_SPEED) #zombies status is turned always while alive
    conn.commit() #commit so it will work
    cur.close()
    conn.close()

def _insert_agent(cur, agent_type, health_status, speed): #gives agents random stats 
    x, y = random.uniform(0, WIDTH), random.uniform(0, HEIGHT) #random spawn
    strength = random.uniform(5, 15) #damage in a fight 
    intelligence = random.uniform(1, 10) #above 5 can work in lab 
    aggression = random.uniform(0, 100) #fight vs flee 
    if agent_type == "human": #for humans only 
        susceptibility = random.uniform(80, 99) #susceptibility starts high
    else:
        susceptibility = 0 #zombies even former humans will have 0 susceptibility cause theyre already zombies 
    cur.execute(
        """INSERT INTO agents
           (agent_type, health_status, x, y, movement_speed, life_value, strength,
            intelligence, susceptibility, aggression)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", #sql give them rolled values 
        (agent_type, health_status, x, y, speed, 100, strength, intelligence, susceptibility, aggression)
                                                
    )

def load_agents():
    """Return every agent row as a list of dicts."""
    conn = get_connection() #new connection 
    cur = conn.cursor()
    cur.execute("""SELECT agent_id, agent_type, health_status, x, y, movement_speed,
                          life_value, strength, intelligence, susceptibility, aggression
                   FROM agents;""") #select all 
    rows = cur.fetchall() #fetch all 
    cur.close()
    conn.close()
    agents = [] 
    for row in rows:
        agents.append({ #ordered dicts bc main zombie needs dicts.
            "agent_id": row[0], "agent_type": row[1], "health_status": row[2],
            "x": row[3], "y": row[4], "movement_speed": row[5],
            "life_value": row[6], "strength": row[7], 
            "intelligence": row[8], "susceptibility": row[9], "aggression": row[10],
        })
    return agents #initiation
