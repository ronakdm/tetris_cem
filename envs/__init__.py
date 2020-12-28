from gym.envs.registration import register

register(
  id='Tetris-v0',
  entry_point='envs.tetris:TetrisEnv'
)
