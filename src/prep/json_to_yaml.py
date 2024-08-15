import json
import yaml

def convert_json_to_yaml(json_file_path):
    with open(json_file_path, 'r') as json_file:
        json_content = json.load(json_file)
    
    # For each key, replace value of `pos` key to int
    for key in json_content.keys():
        json_content[key]['pos'] = int(json_content[key]['pos'])

    yaml_file_path = json_file_path.replace('REF.hla.json', 'hla_info.yaml')
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(json_content, yaml_file, default_flow_style=False)

if __name__ == "__main__":
    filename = "/export/work/users/nonaka/project/HLApj/data/Pan-Asian/Pan-Asian_REF.hla.json"
    convert_json_to_yaml(filename)