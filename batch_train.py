import subprocess
import time
import os
import sys

'''
To run a batch train, set the (algorithm, timesteps, seed) under experiments in line 16.
    Acceptable algorithms: DDPG, SAC

    In terminal: python batch_train.py
'''


print("Batch runner Python:")
print(sys.executable)

# List of experiments to run
experiments = [
    # Algorithm, Timesteps, Seed
    # DDPG runs
    ("DDPG", 100_000, 0),
    #("DDPG", 1_000_000, 1),
    #("DDPG", 1_000_000, 2),


    # SAC runs
    ("SAC", 100_000, 0),
    #("SAC", 1_000_000, 1),
    #("SAC", 1_000_000, 2),
]


# Make log folder if it doesn't exist
os.makedirs("batch_logs", exist_ok=True)


for algorithm, timesteps, seed in experiments:

    print("\n================================")
    print("Starting new experiment")
    print(f"Algorithm: {algorithm}")
    print(f"Timesteps: {timesteps}")
    print(f"Seed: {seed}")
    print("================================\n")

    start_time = time.time()

    log_file = f"batch_logs/{algorithm}_seed{seed}.txt"

    with open(log_file, "w") as log:

        process = subprocess.Popen(
            [
                sys.executable,
                "training.py",
                algorithm,
                str(timesteps),
                str(seed)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="")
            log.write(line)

        process.wait()

    result = process

    elapsed = (time.time() - start_time) / 3600

    if result.returncode == 0:
        print(
            f"Completed {algorithm} seed {seed} "
            f"in {elapsed:.2f} hours"
        )
    else:
        print(
            f"FAILED: {algorithm} seed {seed}. "
            f"Check {log_file}"
        )


print("\n==============================")
print("All experiments finished!")
print("==============================")