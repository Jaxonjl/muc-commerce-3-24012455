import unittest

from app import app


class Day08FlaskTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def login(self):
        return self.client.post(
            "/login",
            data={"username": "student", "password": "day07"},
            follow_redirects=True,
        )

    def test_health_returns_200_and_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/json")
        data = response.get_json()
        self.assertIsInstance(data, dict)
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("service"), "day08-flask-upgrade")

    def test_unauthenticated_metrics_is_rejected(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 401)
        data = response.get_json()
        self.assertEqual(data, {"ok": False, "error": "请先登录后再访问API。"})

    def test_authenticated_metrics_returns_metrics(self):
        self.login()
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("ok"))
        self.assertIsInstance(data.get("metrics"), list)
        self.assertGreater(len(data["metrics"]), 0)
        first_metric = data["metrics"][0]
        self.assertIn("label", first_metric)
        self.assertIn("value", first_metric)
        self.assertIn("note", first_metric)

    def test_categories_filter_fashion_returns_rows(self):
        self.login()
        response = self.client.get("/api/categories?category=Fashion")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("category"), "Fashion")
        self.assertIsInstance(data.get("rows"), list)
        self.assertGreater(len(data["rows"]), 0)
        for row in data["rows"]:
            self.assertEqual(row["偏好品类"], "Fashion")


if __name__ == "__main__":
    unittest.main()
