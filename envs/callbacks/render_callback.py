import time
import mujoco
import mujoco.viewer
from stable_baselines3.common.callbacks import BaseCallback


class LiveRenderCallback(BaseCallback):
    """
    Opens a MuJoCo viewer window and syncs it during training.
    Only renders every `render_freq` steps to avoid slowing training.
    """

    def __init__(self, render_freq: int = 200, slowdown: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.slowdown = slowdown   # seconds to sleep per render (0 = full speed)
        self._viewer = None
        self._mj_model = None
        self._mj_data = None

    def _on_training_start(self) -> None:
        # Unwrap through VecNormalize → DummyVecEnv → AUVDomainRandomWrapper → HalcyonAUVEnv
        env = self.training_env
        while hasattr(env, "venv"):
            env = env.venv
        while hasattr(env, "envs"):
            env = env.envs[0]
        while hasattr(env, "env"):
            env = env.env

        self._mj_model = env.model   # mujoco.MjModel
        self._mj_data  = env.data    # mujoco.MjData

        self._viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)
        print("  🎥  Live viewer launched — drag to rotate, scroll to zoom")

    def _on_step(self) -> bool:
        if self._viewer is None:
            return True
        if not self._viewer.is_running():
            return False   # stop training if viewer is closed
        if self.n_calls % self.render_freq == 0:
            self._viewer.sync()
            if self.slowdown > 0:
                time.sleep(self.slowdown)
        return True

    def _on_training_end(self) -> None:
        if self._viewer:
            self._viewer.close()
