import subprocess
import unittest
from unittest.mock import patch

from pixi_ci import run_install


class PixiTransportTests(unittest.TestCase):
    def setUp(self):
        self.command = ["pixi", "lock", "--manifest-path", "pixi.toml"]

    @patch("pixi_ci.time.sleep")
    @patch("pixi_ci.subprocess.run")
    def test_success_is_not_retried(self, run, sleep):
        run.return_value = subprocess.CompletedProcess(self.command, 0, "installed\n")
        run_install(self.command)
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()

    @patch("pixi_ci.time.sleep")
    @patch("pixi_ci.subprocess.run")
    def test_streaming_wheel_timeout_retries_the_command(self, run, sleep):
        run.side_effect = [subprocess.CompletedProcess(self.command, 1,
                          "Failed to unzip wheel: torch.whl\noperation timed out\n"),
                           subprocess.CompletedProcess(self.command, 0, "installed\n")]
        run_install(self.command)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(5)
        self.assertEqual(run.call_args.args[0], self.command)

    @patch("pixi_ci.time.sleep")
    @patch("pixi_ci.subprocess.run")
    def test_dependency_conflicts_fail_without_retry(self, run, sleep):
        run.return_value = subprocess.CompletedProcess(self.command, 1,
                                                       "No solution found: incompatible package versions\n")
        with self.assertRaises(subprocess.CalledProcessError):
            run_install(self.command)
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()

    @patch("pixi_ci.time.sleep")
    @patch("pixi_ci.subprocess.run")
    def test_repeated_network_failures_are_bounded(self, run, sleep):
        run.return_value = subprocess.CompletedProcess(self.command, 1, "connection reset\n")
        with self.assertRaises(subprocess.CalledProcessError):
            run_install(self.command)
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10])

    def test_invalid_retry_count_fails_before_running_a_process(self):
        with self.assertRaises(ValueError):
            run_install(self.command, attempts=0)


if __name__ == "__main__":
    unittest.main()
