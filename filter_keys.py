import json
import sys

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def extract_unique_keys(data):
    unique_keys = set()
    for category in data.values():
        for result in category.get('results', []):
            unique_keys.add(result.get('unique_key'))
    return unique_keys

def filter_results(data, unique_keys):
    filtered_data = {}
    for category, category_data in data.items():
        filtered_results = [
            result for result in category_data.get('results', [])
            if result.get('unique_key') in unique_keys
        ]
        if filtered_results:
            filtered_data[category] = {'results': filtered_results}
    return filtered_data

def main(file1, file2, output_file):
    # Load JSON data from both files
    data1 = load_json(file1)
    data2 = load_json(file2)

    # Extract unique keys from the first file
    unique_keys = extract_unique_keys(data1)

    # Filter results in the second file based on unique keys
    filtered_data = filter_results(data2, unique_keys)

    # Write the filtered results to the output file
    with open(output_file, 'w') as file:
        json.dump(filtered_data, file, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <file1> <file2> <output_file>")
        sys.exit(1)

    file1, file2, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    main(file1, file2, output_file)
