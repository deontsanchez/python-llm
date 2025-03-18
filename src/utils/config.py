import os
import yaml

# Use arrow function for public API as requested
load_config = lambda config_name: _load_config(config_name)

def _load_config(config_name):
    """
    Load a YAML configuration file from the config directory.
    
    Args:
        config_name (str): Name of the configuration file (with or without .yaml extension)
        
    Returns:
        dict: Configuration as a dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    # Add .yaml extension if not present
    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"
    
    # Construct path to the config file
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
    config_path = os.path.join(config_dir, config_name)
    
    # Check if the file exists
    if not os.path.exists(config_path):
        # Try with example_ prefix if the file doesn't exist
        example_path = os.path.join(config_dir, f"example_{config_name}")
        if os.path.exists(example_path):
            print(f"Warning: {config_name} not found, using example configuration instead.")
            config_path = example_path
        else:
            raise FileNotFoundError(f"Configuration file {config_name} not found in {config_dir}")
    
    # Load the YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_available_configs():
    """
    List all available configuration files.
    
    Returns:
        list: List of configuration file names (without extension)
    """
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
    
    config_files = []
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.yaml'):
                config_files.append(file.replace('.yaml', ''))
    
    return config_files 