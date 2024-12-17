# Import necessary libraries
import yaml

def load_yaml_config(file_path):
    """
    Loads a YAML configuration file and returns the contents as a Python dictionary.

    Args:
    - file_path (str): The path to the YAML configuration file.

    Returns:
    - dict: The contents of the YAML file as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Successfully loaded configuration from {file_path}")
        return config
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except yaml.YAMLError as e:
        print(f"Error while parsing YAML file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def main():
    """
    Main function to demonstrate loading a YAML configuration file.
    """
    # Path to the YAML configuration file (update this path as needed)
    yaml_file_path = 'config.yaml'

    # Load the configuration
    config = load_yaml_config(yaml_file_path)

    # Print the loaded configuration
    if config:
        print("\n🔍 Loaded Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()