import unittest
import os
class TestOutputFolder(unittest.TestCase):
   def test_output_folder_creation(self):
       output_folder = "photo_network"
       if not os.path.exists(output_folder):
           os.makedirs(output_folder)
       self.assertTrue(os.path.exists(output_folder))
if __name__ == '__main__':
   unittest.main()
