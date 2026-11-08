import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE = """
import json
from ai_workers.core.celery_app import app
c = app.conf
print("CONF:" + json.dumps({
    "broker_pool_limit": c.broker_pool_limit,
    "redis_max_connections": c.redis_max_connections,
    "remote_control": c.worker_enable_remote_control,
    "gossip": c.worker_gossip,
    "mingle": c.worker_mingle,
    "events": c.worker_send_task_events,
    "max_retries": c.broker_connection_max_retries,
    "retry_on_startup": c.broker_connection_retry_on_startup,
    "transport": dict(c.broker_transport_options),
}))
"""


def load_conf(**env):
    # Imported in a subprocess: the worker module rewires stdout/stderr, which breaks pytest capture.
    full_env = {k: v for k, v in os.environ.items() if k != "WORKER_REMOTE_CONTROL"}
    full_env.update(env)
    out = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=REPO_ROOT, env=full_env,
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout
    line = next(l for l in out.splitlines() if l.startswith("CONF:"))
    return json.loads(line[len("CONF:"):])


def test_worker_keeps_redis_connection_count_minimal():
    conf = load_conf()
    assert conf["broker_pool_limit"] == 1
    assert conf["redis_max_connections"] == 2
    # No gossip/mingle/event channels: each of them costs another Redis client.
    assert conf["gossip"] is False
    assert conf["mingle"] is False
    assert conf["events"] is False


def test_remote_control_stays_on_by_default_for_gpu_routing():
    # The backend inspects the GPU worker's queues to route jobs to it and revokes cancelled jobs.
    assert load_conf()["remote_control"] is True


def test_container_can_drop_remote_control():
    assert load_conf(WORKER_REMOTE_CONTROL="false")["remote_control"] is False


def test_worker_waits_for_redis_slots_instead_of_giving_up():
    conf = load_conf()
    assert conf["max_retries"] is None
    assert conf["retry_on_startup"] is True
    assert conf["transport"]["socket_keepalive"] is True
    assert conf["transport"]["health_check_interval"] == 30
