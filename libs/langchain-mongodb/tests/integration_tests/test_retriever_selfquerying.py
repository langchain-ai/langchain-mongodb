import os
from time import sleep
from typing import Sequence, Union

import pytest
from flaky import flaky
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from langchain_mongodb.retrievers.self_querying import (
    MongoDBAtlasSelfQueryRetriever,
    MongoDBStructuredQueryTranslator,
)
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "test_self_querying_retriever"


if "OPENAI_API_KEY" not in os.environ:
    pytest.skip("Requires OpenAI for chat responses.", allow_module_level=True)


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
def field_info() -> Sequence[Union[AttributeInfo, dict]]:
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
) -> MongoDBAtlasVectorSearch:
    """Vectorstore without documents nor index."""
    vs = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string,
        namespace=f"{DB_NAME}.{COLLECTION_NAME}",
        embedding=embedding,
    )

    vs.collection.delete_many({})
    if any(ix["name"] == vs._index_name for ix in vs.collection.list_search_indexes()):
        vs.collection.drop_search_index(vs._index_name)
        sleep(5)  # TODO - Follow existing practice

    return vs


@pytest.fixture(scope="module")
def llm():
    """Model used for interpreting query."""
    return ChatOpenAI(model="gpt-4o")


@flaky
def test(vectorstore, llm, field_info, fictitious_movies, dimensions, embedding):
    """Tests MongoDBAtlasSelfQueryRetriever and MongoDBStructuredQueryTranslator
    - Start with a collection of documents.
    - Define your Embedding Model, for search based on symantic similarity.
    - Define AttributeInfo, values that the LLM can use to index and filter on.
    - Create a VectorSearch Index using the model and field indexes
    - Create VectorStore and add Documents


    """

    # Create index along with filter field indexes
    vectorstore.create_vector_search_index(
        dimensions=dimensions,
        filters=[f.name for f in field_info],
        update=False,  # TODO Check if update works on local atlas
        wait_until_complete=60,
    )

    retriever = MongoDBAtlasSelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        metadata_field_info=field_info,
        structured_query_translator=MongoDBStructuredQueryTranslator(),
        document_contents="Descriptions of movies",
        use_original_query=False,  # TODO - set True if we discover why query is thrown out.
        enable_limit=True,  # TODO test cases
        # fix_invalid=True,  # TODO remove if we can. I think the o4 fixes this
        k=10,  # TODO Get this to propagate to _similarity_search_with_score
        search_kwargs={"k": 12},
    )

    assert isinstance(retriever, SelfQueryRetriever)

    # Add documents, including embeddings
    vectorstore.add_documents(fictitious_movies)
    sleep(5)  # TODO - Follow existing practice

    # This example specifies a filter
    res_filter = retriever.invoke("I want to watch a movie rated higher than 8.5")
    assert isinstance(res_filter, list)
    assert isinstance(res_filter[0], Document)
    assert len(res_filter) == 1
    assert res_filter[0].metadata["title"] == "The Coda Paradox"

    # This example specifies a composite AND filter
    res_and = retriever.invoke(
        "Provide movies made after 2030 that are rated lower than 8"
    )
    assert isinstance(res_and, list)
    assert len(res_and) == 2
    assert set(film.metadata["title"] for film in res_and) == {
        "Manifesto Midnight",
        "Neon Tide",
    }

    res_or = retriever.invoke("Provide movies made after 2030 or rated higher than 8.4")
    assert isinstance(res_or, list)
    assert len(res_or) == 4
    assert set(film.metadata["title"] for film in res_or) == {
        "Manifesto Midnight",
        "Neon Tide",
        "The Abyssal Crown",
        "The Coda Paradox",
    }

    # This one does not have a filter
    res_nofilter = retriever.invoke("Provide movies that take place underwater")
    assert len(res_nofilter) == len(fictitious_movies)

    # This example gives a limit
    res_limit = retriever.invoke("Provide 3 movies")
    assert len(res_limit) == 3
