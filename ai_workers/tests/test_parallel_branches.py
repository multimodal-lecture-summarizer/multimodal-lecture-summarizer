from __future__ import annotations

import threading
import time
import unittest

from ai_workers.core.parallel_branches import is_resource_exhaustion, run_branches


class RunBranchesTests(unittest.TestCase):
    def test_sequential_runs_in_declaration_order(self):
        order = []
        results = run_branches(
            [("a", lambda: order.append("a") or 1), ("b", lambda: order.append("b") or 2)],
            parallel=False,
        )
        self.assertEqual(results, {"a": 1, "b": 2})
        self.assertEqual(order, ["a", "b"])

    def test_parallel_overlaps_branches(self):
        barrier = threading.Barrier(2, timeout=5)

        def branch(value):
            def run():
                barrier.wait()  # only passes if both branches are running at once
                return value
            return run

        results = run_branches([("a", branch(1)), ("b", branch(2))], parallel=True)
        self.assertEqual(results, {"a": 1, "b": 2})

    def test_oom_branch_reruns_sequentially_after_the_other_finishes(self):
        events = []
        attempts = {"a": 0}

        def flaky():
            attempts["a"] += 1
            if attempts["a"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
            events.append("a-retry")
            return "a-ok"

        def other():
            time.sleep(0.05)
            events.append("b-done")
            return "b-ok"

        fallbacks, cleanups = [], []
        results = run_branches(
            [("a", flaky), ("b", other)],
            parallel=True,
            on_fallback=lambda name, exc: fallbacks.append(name),
            cleanup=cleanups.append,
        )
        self.assertEqual(results, {"a": "a-ok", "b": "b-ok"})
        self.assertEqual(events, ["b-done", "a-retry"])
        self.assertEqual(fallbacks, ["a"])
        self.assertEqual(len(cleanups), 1)

    def test_non_resource_error_propagates_without_retry(self):
        calls = {"n": 0}

        def broken():
            calls["n"] += 1
            raise ValueError("bad input")

        with self.assertRaises(ValueError):
            run_branches([("a", broken), ("b", lambda: 1)], parallel=True)
        self.assertEqual(calls["n"], 1)

    def test_revocation_propagates(self):
        def revoked():
            raise Exception("Tác vụ bị hủy bởi người dùng.")

        with self.assertRaises(Exception) as ctx:
            run_branches([("a", revoked), ("b", lambda: 1)], parallel=True)
        self.assertIn("hủy", str(ctx.exception))

    def test_fatal_error_sets_abort_and_original_error_wins(self):
        abort = threading.Event()

        def failing():
            raise ValueError("root cause")

        def sibling():
            abort.wait(timeout=5)  # stops as soon as the sibling failure is signalled
            raise Exception("aborted because sibling failed")

        with self.assertRaises(ValueError):
            run_branches([("a", sibling), ("b", failing)], parallel=True, abort=abort)
        self.assertTrue(abort.is_set())

    def test_resource_exhaustion_detection(self):
        self.assertTrue(is_resource_exhaustion(MemoryError()))
        self.assertTrue(is_resource_exhaustion(RuntimeError("CUDA out of memory")))
        self.assertTrue(is_resource_exhaustion(RuntimeError("Insufficient RAM to start")))
        self.assertFalse(is_resource_exhaustion(ValueError("nope")))


if __name__ == "__main__":
    unittest.main()
