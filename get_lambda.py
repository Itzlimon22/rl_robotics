import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_final_lambda(seed_dir):
    tb_path = os.path.join(seed_dir, "tensorboard")
    if not os.path.exists(tb_path):
        return None
    subdirs = os.listdir(tb_path)
    if not subdirs:
        return None
    event_file = os.path.join(tb_path, subdirs[0])
    ea = EventAccumulator(event_file)
    ea.Reload()
    try:
        events = ea.Scalars("cdr/curriculum_level")
        if events:
            return events[-1].value
    except:
        pass
    return None

print("seed 0:", get_final_lambda("/Users/limon/rl_research/auv/master_curriculum/master_curriculum_seed0"))
print("seed 1:", get_final_lambda("/Users/limon/rl_research/auv/master_curriculum/master_curriculum_seed1"))
print("seed 0_v1:", get_final_lambda("/Users/limon/rl_research/auv/master_curriculum/master_curriculum_seed0_v1"))
