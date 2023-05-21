import unittest
from unittest.mock import patch
from run import Singleton, GenrePredictor, create_app 

class TestSingleton(unittest.TestCase):
    def test_singleton(self):
        class SingletonClass(metaclass=Singleton):
            pass

        instance1 = SingletonClass()
        instance2 = SingletonClass()
        
        self.assertEqual(id(instance1), id(instance2))


class TestGenrePredictor(unittest.TestCase):
    @patch.object(GenrePredictor, 'model', return_value=None)
    def test_classify_genre(self, mock_model):
        mock_model.get_predictions.return_value = ["Drama", "Comedy"]
        genre_predictor = GenrePredictor()
        result = genre_predictor.classify_genre("Some text")
        
        self.assertEqual(result, ["Drama", "Comedy"])
        mock_model.get_predictions.assert_called_once_with("Some text")

    @patch.object(GenrePredictor, 'model', return_value=None)
    def test_post(self, mock_model):
        app = create_app()
        with app.test_client() as client:
            mock_model.get_predictions.return_value = ["Drama", "Comedy"]
            # Test with a valid overview
            response = client.post('/', data=dict(overview="Action packed movie"))
            self.assertEqual(response.status_code, 200)

            # Test without overview
            response = client.post('/', data=dict())
            self.assertEqual(response.status_code, 400)

            # Test with invalid model prediction (mock the error)
            with patch.object(GenrePredictor, "classify_genre", side_effect=Exception("Error")):
                response = client.post('/', data=dict(overview="Some text"))
                self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main()
