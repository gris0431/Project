import unittest
from observer.py import cosine_similarity, get_hog_descriptor

class TestVideoSurveillance(unittest.TestCase):

    def test_cosine_similarity(self):
        # Тест на проверку вычисления косинусного сходства
        vec_a = np.array([1, 2, 3])
        vec_b = np.array([4, 5, 6])
        result = cosine_similarity(vec_a, vec_b)
        expected = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        self.assertAlmostEqual(result, expected)

    def test_get_hog_descriptor_shape(self):
        # Тест на проверку размера HOG дескриптора
        image = np.random.rand(128, 64, 3).astype('uint8')
        descriptor = get_hog_descriptor(image)
        self.assertEqual(descriptor.shape[0], 3780)  # 3780 - стандартный размер для HOG дескриптора

    def test_get_hog_descriptor_values(self):
        # Тест на проверку значений HOG дескриптора
        image = np.random.rand(128, 64, 3).astype('uint8')
        descriptor = get_hog_descriptor(image)
        self.assertTrue(np.all(descriptor >= 0))  # Все значения должны быть неотрицательными

    # Дополнительные тесты могут включать проверку исключений, граничных условий и т.д.

if name == 'main':
    unittest.main()
