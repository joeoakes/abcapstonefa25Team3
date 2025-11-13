import unittest
import sys
from pathlib import Path

def run_tests():
    """Run all tests in the tests directory"""
    # Add the parent directory to Python path to allow importing project modules
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = str(Path(__file__).parent)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Custom test result class to add line breaks between test files
    class VerboseResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.current_module = None
        
        def startTest(self, test):
            module = test.__class__.__module__
            if self.current_module and self.current_module != module:
                print("\n" + "="*70)
            self.current_module = module
            super().startTest(test)
    
    runner = unittest.TextTestRunner(verbosity=2, resultclass=VerboseResult)
    result = runner.run(suite)
    
    # Return 0 if tests passed, 1 if any failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())