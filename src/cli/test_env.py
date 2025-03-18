#!/usr/bin/env python3
from src.utils.environment import test_environment

def main():
    """Run environment tests and display results"""
    results = test_environment()
    
    # You could add additional reporting or checks here
    # based on the results dictionary

if __name__ == "__main__":
    main() 