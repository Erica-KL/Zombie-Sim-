import contextlib, copy, io, os, random, subprocess, sys, time #standard tools
os.environ.setdefault("ZOMBIE_AGENTS", "800") #simulation.py reads this at import (600 humans + 200 zombies, quick to test)
import pytest #the test runner
import simulation, zombie_storage as zs #the code being tested

def crowd(seed=1, box=250): #everyone squeezed into a small box so there are plenty of contacts
    random.seed(seed) #same crowd every time
    agents = zs.Simulation.new_state()["agents"] #the normal starting population
    for a in agents: a["x"], a["y"] = random.uniform(0, box), random.uniform(0, box) #move everyone into the box
    return agents #the crowd

@pytest.fixture #set up for a test
def pool(monkeypatch): #use the worker pool even though the population is small
    monkeypatch.setattr(simulation, "PAR_MIN_AGENTS", 0) #no minimum agent count
    monkeypatch.setattr(simulation, "PAR_MIN_COST", 0) #no minimum work
    monkeypatch.delenv("ZOMBIE_START_METHOD", raising=False) #default start method
    if simulation.fork_context() is None: pytest.skip("this machine cannot fork workers") #skip where workers cannot start

def running(store): #a simulation that is not paused
    random.seed(5) #same start every time
    sim = zs.Simulation(store); sim.toggle_pause() #new simulation, unpaused
    return sim #ready to step

# ---------------------------------------------------------------- the rules
def test_default_population_is_750_humans_and_250_zombies(): #checks the default start
    env = {k: v for k, v in os.environ.items() if k != "ZOMBIE_AGENTS"} #settings without an agent count
    out = subprocess.run([sys.executable, "-c", "import simulation as s; print(s.NUM_HUMANS, s.NUM_ZOMBIES)"], #ask a fresh python for the counts
                         capture_output=True, text=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__))) #run it in this folder
    assert out.stdout.split() == ["750", "250"], out.stderr #750 humans, 250 zombies

def test_contacts_match_checking_every_single_pair(): #the fast search against the slow sure one
    agents = crowd() #a crowd
    damage, first = simulation.process_contacts(agents, 0.1, 1) #the fast way, 1 core
    touching = {} #human index -> zombies within reach, found the slow obvious way
    for h, a in enumerate(agents): #every agent as the human
        for z, b in enumerate(agents): #every agent as the zombie
            if (a["agent_type"], b["agent_type"]) == ("human", "zombie") and "dead" not in (a["health_status"], b["health_status"]) \
                    and (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 <= simulation.COLLISION_RADIUS ** 2: #and within touch distance
                touching.setdefault(h, []).append(z) #they touch
    assert touching and first == {h: min(zs_) for h, zs_ in touching.items()} #the lowest zombie index is the one that bites
    for h, zs_ in touching.items(): #a human takes the summed hits of every zombie touching it
        assert damage[h] == sum(round(agents[z]["strength"] * simulation.ATTACK_SCALE * 0.1 * simulation.FIX) for z in zs_) #same sum as adding by hand

def test_nearest_zombie_matches_checking_everyone(): #the fast lookup against the slow sure one
    agents = crowd(seed=2, box=600) #a looser crowd
    index = simulation.Perception(agents) #the index
    for a in agents[:100]: #first 100 agents
        seen = [(( a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2, i) for i, b in enumerate(agents) if b["agent_type"] == "zombie"] #distance to every zombie
        d2, i = min(seen) #the closest one
        assert index.nearest_zombie(a) is (agents[i] if d2 <= simulation.PERCEPTION_RADIUS ** 2 else None) #same agent, or nobody when out of sight

def test_one_patient_gets_one_dose_even_with_two_carriers_in_reach(): #no double dosing
    agents = crowd() #a crowd
    for i, x in ((0, 1000.0), (1, 1002.0), (2, 1001.0)): agents[i].update(x=x, y=1000.0) #three humans alone, far from the crowd
    agents[0]["medicine_count"] = agents[1]["medicine_count"] = 1 #two carriers
    agents[2]["health_status"] = "infected" #one patient
    with contextlib.redirect_stdout(io.StringIO()): simulation.deliver_medicine(agents, 1) #deliver, quietly
    assert agents[2]["health_status"] == "healthy" and agents[0]["medicine_count"] + agents[1]["medicine_count"] == 1 #cured, and exactly one dose used

# ---------------------------------------------------------------- parallelism
def final_state(workers, ticks=3): #a short full run, returns every agent at the end
    agents = crowd(seed=3) #same crowd for every core count
    simulation.WORKERS = workers #cores to use
    with contextlib.redirect_stdout(io.StringIO()): #quietly
        for tick in range(ticks): #every tick
            random.seed(tick) #same dice every run
            for _ in range(zs.STEPS_PER_TICK): #10 steps
                simulation.step(agents, dt=zs.STEP, night=False) #move
                simulation.check_collisions(agents, tick, dt=zs.STEP) #fights
            simulation.check_bites(agents, tick + 1) #bites
            simulation.advance_infections(agents); simulation.advance_exposure(agents) #timers
            simulation.deliver_medicine(agents, tick + 1); simulation.advance_research(agents, tick + 1) #cures and lab work
            simulation.decide_stances(agents) #fight or flee
    simulation.WORKERS = 1 #back to 1 core for other tests
    return agents #everyone at the end

def test_the_final_state_is_identical_on_1_2_and_4_cores(pool): #the main promise of the design
    one = final_state(1) #1 core
    assert any(a["health_status"] == "dead" for a in one) #fights really happened
    assert final_state(2) == one and final_state(4) == one #2 and 4 cores end exactly the same

def _die(x): os._exit(1) #a worker killed from the inside
def _hang(x): time.sleep(5) #a worker that never answers

def test_a_dead_or_hung_worker_does_not_freeze_the_run(pool, monkeypatch): #worker failures
    assert simulation.run_parallel(_die, [1, 2, 3, 4], 2) is None #None tells the caller to do the work itself
    monkeypatch.setattr(simulation, "POOL_TIMEOUT", 0.5) #only wait half a second
    start = time.time() #start the clock
    assert simulation.run_parallel(_hang, [1, 2], 2) is None and time.time() - start < 4 #gave up quickly instead of waiting

def test_when_the_pool_fails_the_contacts_are_still_the_same(pool, monkeypatch): #fallback gives the same answer
    agents = crowd() #a crowd
    good = simulation.process_contacts(agents, 0.1, 1) #the answer on 1 core
    monkeypatch.setattr(simulation, "run_parallel", lambda *a, **k: None) #the pool "failed"
    assert simulation.process_contacts(agents, 0.1, 4) == good #4 cores with a broken pool, same answer

# ---------------------------------------------------------------- bad data and crashes
@pytest.mark.parametrize("field, value", [("x", float("nan")), ("x", -50.0), ("life_value", float("inf")), ("health_status", "sleepy")]) #four kinds of bad data
def test_bad_data_is_caught(field, value): #each one must be noticed
    state = zs.Simulation.new_state() #good data
    assert simulation.check_state(state) == [] #no problems
    state["agents"][0][field] = value #break one value
    assert simulation.check_state(state) #now there is a problem

def test_a_crashing_step_is_undone_and_the_simulation_pauses(monkeypatch): #undo and pause
    sim = running(zs.InMemoryStore()) #running, in memory
    before = copy.deepcopy(sim.state["agents"]) #copy of everyone before
    monkeypatch.setattr(simulation, "check_collisions", lambda *a, **k: 1 / 0) #make the step crash
    sim.advance() #must not raise
    assert sim.state["agents"] == before and sim.status()["paused"] and "ZeroDivisionError" in sim.status()["error"] #everyone unchanged, paused, error shown

# ---------------------------------------------------------------- the HDF5 file
h5py = pytest.importorskip("h5py") #skip these when h5py is missing

def test_what_is_saved_is_exactly_what_comes_back(tmp_path): #save then load
    random.seed(3) #same population every time
    state = zs.Simulation.new_state() #a starting state
    store = zs.HDF5Store(str(tmp_path / "s.h5")); store.save(state) #save it
    assert store.load() == state #loads back exactly

def test_a_damaged_file_gives_a_clear_error_not_a_silent_fresh_start(tmp_path): #a broken file
    store = zs.HDF5Store(str(tmp_path / "s.h5")); store.save(zs.Simulation.new_state()) #a good save
    open(store.path, "wb").write(b"junk") #wreck the file
    with pytest.raises(zs.StoreCorrupt): zs.Simulation(store, resume=True) #resume must stop with the named error

def test_one_save_per_tick_and_a_killed_run_resumes_from_the_last_tick(tmp_path, monkeypatch): #saving and resuming
    path = str(tmp_path / "s.h5"); store = zs.HDF5Store(path); sim = running(store) #a running simulation on a file
    saves, real = [], store.save #count the saves
    monkeypatch.setattr(store, "save", lambda s: (saves.append(1), real(s))[1]) #count, then really save
    with contextlib.redirect_stdout(io.StringIO()): #quietly
        for _ in range(15): sim.advance() #one and a half ticks, then the process "dies"
    assert len(saves) == 1 #15 steps, 1 save
    back = zs.Simulation(zs.HDF5Store(path), resume=True) #start again from the file
    assert back.state["runtime"]["tick"] == 1 and back.state["runtime"]["steps"] == 0 #back at the end of tick 1
