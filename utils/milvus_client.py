# utils/milvus_client.py

from pymilvus import connections


def connect_milvus(host="localhost", port="19530"):
    connections.connect(alias="default", host=host, port=port)
    print("Connected to Milvus")


if __name__ == "__main__":
    connect_milvus()
