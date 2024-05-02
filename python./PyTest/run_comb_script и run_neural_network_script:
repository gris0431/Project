import unittest
from unittest.mock import patch
from io import StringIO
import sys
class TestScripts(unittest.TestCase):
   @patch('sys.stdout', new_callable=StringIO)
   def test_run_comb_script(self, mock_stdout):
       run_comb_script()
       self.assertEqual(mock_stdout.getvalue(), )
   @patch('sys.stdout', new_callable=StringIO)
   def test_run_neural_network_script(self, mock_stdout):
       with self.assertRaises(KeyboardInterrupt):
           run_neural_network_script()
