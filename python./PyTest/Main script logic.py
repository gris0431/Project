import unittest
from unittest.mock import patch, MagicMock, call
import concurrent.futures
class TestScriptExecution(unittest.TestCase):
   @patch('concurrent.futures.ThreadPoolExecutor', autospec=True)
   def test_script_execution(self, mock_executor):
       mock_future_comb = MagicMock()
       mock_future_neural_network = MagicMock()
       mock_executor.return_value.__enter__.return_value.submit.side_effect = [mock_future_comb, mock_future_neural_network]
       with patch('concurrent.futures.wait') as mock_wait, patch('builtins.print') as mock_print:
           mock_future_comb.done.return_value = True
           mock_wait.return_value = (mock_future_comb,)
           run_comb_script()
           run_neural_network_script()
           mock_future_comb.cancel.assert_called_once()
           mock_future_neural_network.cancel.assert_called_once()
           mock_print.assert_called_with('Both scripts have finished executing')
if __name__ == '__main__':
    unittest.main()
