"""Passive substep pose recording for pinned Safety-Gymnasium 1.0.

Every mj_step call is delegated unchanged; only the selected model/data is
recorded. The original function is restored on every exit, including failure.
"""
from contextlib import contextmanager
from threading import RLock

import numpy as np

_TRACE_LOCK = RLock()


@contextmanager
def capture_native_motion(model, data, body_name="agent"):
    import mujoco
    body_id = model.body(body_name).id
    with _TRACE_LOCK:
        original = mujoco.mj_step
        points = [np.array(data.xpos[body_id, :2]).tolist()]

        def recorded_step(step_model, step_data, *args, **kwargs):
            selected = step_model is model and step_data is data
            if selected and (args or kwargs.get("nstep", 1) != 1):
                raise RuntimeError("substep tracing requires the pinned single-step backend")
            result = original(step_model, step_data, *args, **kwargs)
            if selected:
                points.append(np.array(data.xpos[body_id, :2]).tolist())
            return result

        mujoco.mj_step = recorded_step
        try:
            yield points
        finally:
            mujoco.mj_step = original
            # The backend's final mj_forward refreshes the final body pose.
            points.append(np.array(data.xpos[body_id, :2]).tolist())
