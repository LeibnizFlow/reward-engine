"""Stop new task dispatch on resource exhaustion without cancelling live tasks."""

import shutil


def check_space(config):
    minimum = config.get("minimum_free_bytes", 5 * 1024**3)
    if type(minimum) is not int or minimum < 0:
        raise ValueError("minimum_free_bytes must be a nonnegative integer")
    for directory in (config["output"], config.get("docker_data_root", "/var/lib/docker")):
        available = shutil.disk_usage(directory).free
        if available < minimum:
            raise OSError(
                f"Insufficient free space to dispatch another trial: {directory}; "
                f"available={available}, required={minimum}"
            )


class TaskAdmission:
    """Legacy runs latch disk failures; isolated scheduling rechecks after recovery."""

    def __init__(self, config, check):
        self.config = config
        self.check = check
        self.failure = None

    def admit(self):
        if self.failure is not None and self.config.get("failure_policy") != "isolate_tasks_v1":
            raise OSError(self.failure)
        try:
            self.check(self.config)
            self.failure = None
        except OSError as error:
            self.failure = str(error)
            raise
