import subprocess
import sys

TRAIN_OPERATIONS = ["prepare_data", "train_reader", "tune_embedder"]
QUERY_OPERATIONS = ["query_reader", "query_retriever"]

def run(operation, count = None, query = None):
    command = None
    if count is None and TRAIN_OPERATIONS.__contains__(operation):
        command = f"python -m scripts.{operation}"
    elif count is not None and QUERY_OPERATIONS.__contains__(operation):
        command = f"python -m scripts.{operation} {count} {query}"

    if command is not None:
        subprocess.call([command], shell=True)

def main():
    if len(sys.argv) < 2:
        raise Exception("Operation parameter missing")
    elif len(sys.argv) == 3:
        raise Exception("Query parameter missing")
    else:
        operation = sys.argv[1]
        if len(sys.argv) > 3:
            count = sys.argv[2]
            query = ' '.join(sys.argv[3:])
        else:
            count, query = None, None
    
    run(operation, count, query)

if __name__ == '__main__':
    main()
