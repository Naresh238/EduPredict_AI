
import unittest

from app import app


class EduPredictAppTests(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_analytics_page(self):
        response = self.client.get("/analytics")
        self.assertEqual(response.status_code, 200)

    def test_about_page(self):
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)

    def test_prediction_page(self):
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
