from langchain_mongodb import __all__

EXPECTED_ALL = [
    "MongoDBChatMessageHistory",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
