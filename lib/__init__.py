from gym.envs.registration import register


MAX_EPISODE_STEPS = 1e5

register(
    id='GridWorld-v0',
    entry_point='lib.envs.grid_world:GridWorld',
    tags={'wrapper_config.TimeLimit.max_episode_steps': MAX_EPISODE_STEPS},
)

register(
    id='WindyGridWorld-v0',
    entry_point='lib.envs.windy_grid_world:WindyGridworld',
    tags={'wrapper_config.TimeLimit.max_episode_steps': MAX_EPISODE_STEPS},
)
