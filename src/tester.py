from retriever import Vectorizer
import json

def main():
    with open("./data/train-v1.1.json", "r") as f:
        squad = json.load(f)
    
    print(squad['version'])
    print(squad['data'][0].keys())
    print(len(squad['data'][0]['paragraphs']))
    print(squad['data'][0]['paragraphs'][0])
    print(len(squad['data']))

if __name__ == '__main__':
    main()
