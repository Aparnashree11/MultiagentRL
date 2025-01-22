from pettingzoo.atari import boxing_v2

rom_path = "roms"
env = boxing_v2.parallel_env()
env.reset(seed=42)
print("Environment initialized successfully!")
env.close()
