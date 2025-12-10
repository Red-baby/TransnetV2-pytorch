import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='List 0-based indices where labels==1 and many_hot_labels==1.')
    parser.add_argument('--json-path', required=True, help='Path to JSON file with labels/many_hot_labels array.')
    parser.add_argument('--show-many-hot', action='store_true', help='Also list indices where many_hot_labels==1.')
    args = parser.parse_args()

    with open(args.json_path, 'r', encoding='utf-8') as f:
        entry = json.load(f)
    labels = entry['labels']
    label_ones = [i for i, v in enumerate(labels) if v == 1]
    for idx in label_ones:
        print(idx)

    if args.show_many_hot:
        many_hot = entry.get('many_hot_labels', [])
        mh_ones = [i for i, v in enumerate(many_hot) if v == 1]
        print("# many_hot indices:")
        for idx in mh_ones:
            print(idx)

if __name__ == '__main__':
    main()
