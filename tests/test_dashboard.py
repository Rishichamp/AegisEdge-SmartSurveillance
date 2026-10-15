import base64
import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualization.dashboard import DashboardServer  # noqa: E402


def _basic(user, password):
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


class TestDashboardAuth(unittest.TestCase):
    def setUp(self):
        os.environ["AEGIS_DASHBOARD_PASSWORD"] = "s3cret"
        self.server = DashboardServer(host="127.0.0.1", port=0)
        self.client = self.server.app.test_client()

    def tearDown(self):
        os.environ.pop("AEGIS_DASHBOARD_PASSWORD", None)

    def test_requires_login_when_password_set(self):
        for path in ("/", "/api/status", "/app.js", "/video_feed"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 401, path)
            self.assertIn("Basic", r.headers.get("WWW-Authenticate", ""))
        self.assertEqual(self.client.post("/api/trigger_test").status_code, 401)

    def test_wrong_and_right_password(self):
        self.assertEqual(self.client.get("/api/status", headers=_basic("admin", "nope")).status_code, 401)
        self.assertEqual(self.client.get("/api/status", headers=_basic("admin", "s3cret")).status_code, 200)


class TestDashboardOpenByDefault(unittest.TestCase):
    def test_no_password_means_no_login(self):
        os.environ.pop("AEGIS_DASHBOARD_PASSWORD", None)
        server = DashboardServer(host="127.0.0.1", port=0)
        self.assertEqual(server.app.test_client().get("/api/status").status_code, 200)


class TestFrameBroadcast(unittest.TestCase):
    def test_every_viewer_gets_every_frame(self):
        os.environ.pop("AEGIS_DASHBOARD_PASSWORD", None)
        server = DashboardServer(host="127.0.0.1", port=0)
        server.running = True
        viewers = [server._generate_frames() for _ in range(2)]
        received = [[], []]

        def watch(i):
            for _ in range(3):
                received[i].append(next(viewers[i]))

        threads = [threading.Thread(target=watch, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for k in range(3):
            time.sleep(0.2)
            server.update_frame(np.full((8, 8, 3), k * 40, dtype=np.uint8))
        for t in threads:
            t.join(timeout=5)
        server.running = False
        # both viewers saw all three frames (with the old size-1 queue they split them)
        self.assertEqual([len(r) for r in received], [3, 3])
        self.assertEqual(received[0], received[1])


if __name__ == "__main__":
    unittest.main()
