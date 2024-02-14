import random
import string


def replace_periods_with_underscores(input_string):
    # Use the replace method to replace periods with underscores
    modified_string = input_string.replace('.', '_')
    return modified_string


def generate_random_string(length=5):
    # Define the characters to choose from
    characters = string.ascii_letters + string.digits  # Alphanumeric characters

    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))
    
    return random_string


if __name__ == '__main__':
    for i in range(1000):
        print(generate_random_string())