import os

import pytest

# Set by scripts/start_local_atlas.sh, whose container registers no embedding
# model.  Deployments that support auto-embedding leave it unset, so a new one
# runs these tests by default rather than silently opting out of them.
AUTOEMBED_UNSUPPORTED = "AUTOEMBED_UNSUPPORTED"
AUTOEMBED_MODEL = "voyage-4"


@pytest.fixture(scope="session")
def autoembedding_or_skip() -> None:
    """Skip the requesting test unless the deployment supports auto-embedding."""
    if os.environ.get(AUTOEMBED_UNSUPPORTED):
        pytest.skip(
            f"Deployment has no '{AUTOEMBED_MODEL}' embedding model registered, "
            "so autoEmbed indexes cannot be created"
        )
