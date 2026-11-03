import unittest
from pathlib import Path

import joblib
import pandas as pd


class EduPredictModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models_dir = Path(__file__).resolve().parent.parent / "models"

        cls.grade_model = joblib.load(
            models_dir / "uci_best_grade_model.joblib"
        )
        cls.risk_model = joblib.load(
            models_dir / "uci_best_risk_model.joblib"
        )

        cls.student = pd.DataFrame([{
            "school": "GP",
            "sex": "F",
            "age": 17,
            "address": "U",
            "famsize": "GT3",
            "Pstatus": "T",
            "Medu": 2,
            "Fedu": 2,
            "Mjob": "services",
            "Fjob": "other",
            "reason": "course",
            "guardian": "mother",
            "traveltime": 1,
            "studytime": 2,
            "failures": 0,
            "schoolsup": "no",
            "famsup": "yes",
            "paid": "no",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
            "famrel": 4,
            "freetime": 3,
            "goout": 3,
            "Dalc": 1,
            "Walc": 2,
            "health": 3,
            "absences": 4,
        }])

    def test_grade_model_loads(self):
        self.assertIsNotNone(self.grade_model)

    def test_risk_model_loads(self):
        self.assertIsNotNone(self.risk_model)

    def test_student_has_30_features(self):
        self.assertEqual(self.student.shape[1], 30)

    def test_grade_prediction_is_valid(self):
        prediction = float(self.grade_model.predict(self.student)[0])

        self.assertTrue(
            0 <= prediction <= 20,
            f"Grade prediction outside expected range: {prediction}"
        )

    def test_risk_prediction_is_valid(self):
        prediction = self.risk_model.predict(self.student)[0]

        self.assertIn(int(prediction), [0, 1])

    def test_risk_probability_is_valid(self):
        probabilities = self.risk_model.predict_proba(self.student)[0]

        self.assertAlmostEqual(float(sum(probabilities)), 1.0, places=5)
        self.assertTrue(
            all(0 <= float(probability) <= 1 for probability in probabilities)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
