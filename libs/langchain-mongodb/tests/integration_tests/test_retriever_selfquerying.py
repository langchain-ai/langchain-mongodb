import pytest
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.documents import Document

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "test_self_querying_retriever"


@pytest.fixture(scope="module")
def fictitious_movies():
    """A list of documents that a typical LLM would not know without RAG"""

    return [
        Document(
            page_content="A rogue AI starts producing poetry so profound that it destabilizes global governments.",
            metadata={
                "title": "The Algorithmic Muse",
                "year": 2027,
                "rating": 8.1,
                "genre": "science fiction",
            },
        ),
        Document(
            page_content="A washed-up detective in a floating city stumbles upon a conspiracy involving time loops and missing memories.",
            metadata={
                "title": "Neon Tide",
                "year": 2034,
                "rating": 7.8,
                "genre": "thriller",
            },
        ),
        Document(
            page_content="A group of deep-sea explorers discovers an ancient civilization that worships a colossal, sentient jellyfish.",
            metadata={
                "title": "The Abyssal Crown",
                "year": 2025,
                "rating": 8.5,
                "genre": "adventure",
            },
        ),
        Document(
            page_content="An interstellar chef competes in a high-stakes cooking tournament where losing means permanent exile to a barren moon.",
            metadata={
                "title": "Cosmic Cuisine",
                "year": 2030,
                "rating": 7.9,
                "genre": "comedy",
            },
        ),
        Document(
            page_content="A pianist discovers that every song he plays alters reality in unpredictable ways.",
            metadata={
                "title": "The Coda Paradox",
                "year": 2028,
                "rating": 8.7,
                "genre": "drama",
            },
        ),
        Document(
            page_content="A medieval kingdom is plagued by an immortal bard who sings forbidden songs that rewrite history.",
            metadata={
                "title": "The Ballad of Neverend",
                "year": 2026,
                "rating": 8.2,
                "genre": "fantasy",
            },
        ),
        Document(
            page_content="A conspiracy theorist wakes up to find that every insane theory he's ever written has come true overnight.",
            metadata={
                "title": "Manifesto Midnight",
                "year": 2032,
                "rating": 7.6,
                "genre": "mystery",
            },
        ),
    ]


@pytest.fixture(scope="module")
def metadata_field_info():
    return [
        AttributeInfo(
            name="genre",
            description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating", description="A 1-10 rating for the movie", type="float"
        ),
    ]


@pytest.fixture(scope="module")
def vectorstore(
    connection_string,
    embedding,
    fictitious_movies,
    dimensions,
):
    store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string,
        namespace=f"{DB_NAME}.{COLLECTION_NAME}",
        embedding=embedding,
    )
    store.add_documents(fictitious_movies)
    return store


def test(vectorstore, dimensions):
    """
    - Start with a collection of documents.
    - Define your Embedding Model, for search based on symantic similarity.
    - Define AttributeInfo, values that one can use to index and filter on.
    - Create a VectorSearch Index using the model and field indexes
    -

    Args:
        vectorstore:

    Returns:

    """

    # Steps
    #
    vectorstore.create_vector_search_index(
        dimensions=dimensions, wait_until_complete=60
    )

    pass


# TODO NEXT
#   - Add mappings to vector-search-index
#       * https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/#about-the-filter-type
#       * Take these from list of AttributeInfo. We'll need a function for this
#   - API  add_documents, add_attribute_info (good name?),
