import argparse, contextlib, datetime, hashlib, io, json, os, platform, subprocess, sys, time #standard tools

ap = argparse.ArgumentParser() #reads the command line
ap.add_argument("--agents", type=int, default=1000) #1000 = 750 humans + 250 zombies
ap.add_argument("--workers", type=int, default=1) #cpu cores used
ap.add_argument("--ticks", type=int, default=3) #ticks per run
ap.add_argument("--table", action="store_true") #run every experiment and write results.md
ap.add_argument("--sizes", default="1000,2000,5000,10000,20000") #agent counts for --table
ap.add_argument("--reps", type=int, default=3) #times each row is repeated, the table shows the median
ap.add_argument("--max-workers", type=int, default=os.cpu_count() or 2) #never start more workers than this
args = ap.parse_args() #read them
HERE = os.path.dirname(os.path.abspath(__file__)) #this folder


def run_one(n, workers, ticks): #one timed run
    os.environ["ZOMBIE_AGENTS"] = str(n) #simulation.py reads this when it is imported
    import random, simulation #imported here, after the agent count is set
    import zombie_storage as zs #builds the starting population in memory
    simulation.WORKERS = workers #after the import, zombie_storage sets it too
    random.seed(42) #same start every run
    state = zs.Simulation.new_state() #same population the server starts with
    agents, rt = state["agents"], state["runtime"] #shortcuts
    tick, t0 = 0, time.perf_counter() #tick counter, start the clock
    with contextlib.redirect_stdout(io.StringIO()): #simulation prints every event; silence it
        for _ in range(ticks): #every tick
            random.seed(tick) #same dice for a tick on every run
            for _ in range(zs.STEPS_PER_TICK): #10 steps
                simulation.step(agents, dt=zs.STEP, night=simulation.is_night(tick)) #move
                simulation.check_collisions(agents, tick, dt=zs.STEP) #fights
            tick += 1 #tick done
            simulation.check_bites(agents, tick) #bites
            simulation.advance_infections(agents) #infected -> zombie
            simulation.advance_exposure(agents) #exposed -> healthy
            simulation.deliver_medicine(agents, tick) #cures
            simulation.advance_research(agents, tick) #lab work
            simulation.decide_stances(agents) #fight or flee
    secs = time.perf_counter() - t0 #stop the clock
    digest = hashlib.sha256(repr([(a["health_status"], round(a["x"], 6), round(a["y"], 6)) for a in agents]).encode()).hexdigest()[:12] #fingerprint of the final state
    return dict(agents=n, workers=workers, ticks=ticks, seconds=round(secs, 3), s_per_tick=round(secs / ticks, 3), digest=digest) #the result


def sub(n, w, t): #each run in a fresh process because simulation.py reads the agent count at import
    cmd = [sys.executable, os.path.abspath(__file__), "--agents", str(n), "--workers", str(w), "--ticks", str(t)] #this file, with the settings
    try: #run it
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=HERE) #a run is stopped after 15 minutes
        return json.loads(p.stdout.strip().splitlines()[-1]) #the last line printed is the result
    except subprocess.TimeoutExpired: return {"failed": "stopped after 900 s"} #too slow
    except Exception: return {"failed": "crashed: " + (p.stderr.strip().splitlines() or ["no output"])[-1][:120]} #crashed, keep the last error line


def med(n, w, t): #repeat a run and keep the middle time, so one slow run cannot skew the row
    runs = [sub(n, w, t) for _ in range(max(1, args.reps))] #repeat
    bad = [r for r in runs if "failed" in r] #failed ones
    return bad[0] if bad else sorted(runs, key=lambda r: r["seconds"])[len(runs) // 2] #a failure, or the middle time


if not args.table: #one run
    print(json.dumps(run_one(args.agents, args.workers, args.ticks))) #print its result
else: #the table
    sizes = [int(x) for x in args.sizes.split(",")] #agent counts
    cores = sorted({1, min(os.cpu_count() or 1, args.max_workers)}) #1 core and every core this machine has
    print(f"running {len(sizes) * len(cores) * args.reps} timed runs, results print at the bottom when it's done...", flush=True) #progress
    out = [f"# Stage 3 benchmark ({datetime.datetime.now():%Y-%m-%d %H:%M}, {platform.platform()}, {os.cpu_count()} CPUs, Python {platform.python_version()})",
           f"{args.ticks} ticks per run, 10 steps per tick, middle time of {args.reps} runs\n", #second header line
           "| agents | cores | time (s) |", "|---|---|---|"] #table head
    for n in sizes: #every size
        for w in cores: #1 core, then all cores
            r = med(n, w, args.ticks) #measure
            out.append(f"| {n:,} | {w} | {r['failed'] if 'failed' in r else format(r['seconds'], '.2f')} |") #one row
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(out) + "\n") #save
    print("\n" + "\n".join(out) + "\n\n(also saved to results.md)") #print at the bottom
