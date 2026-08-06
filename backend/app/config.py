import os

import sklearn
from dotenv import load_dotenv

load_dotenv()

# How much memory scikit-learn may use for one chunk of a chunked pairwise
# computation — KNN's distance matrix, silhouette, and anything else built on
# pairwise_distances_chunked.
#
# The default is 1024 MB, which is twice the entire free-tier container. sklearn
# is not being reckless; it simply assumes a machine that has that to spare, and
# sizes its chunks to fill the allowance. Left alone, a KNN fit on a modest
# dataset reserves more memory than the process is allowed to have and the
# container is killed. Smaller chunks compute exactly the same answer, just in
# more passes.
sklearn.set_config(working_memory=int(os.getenv("SKLEARN_WORKING_MEMORY_MB", "64")))

APP_TITLE = "DataForge AI Backend"

# Origins allowed to call this API. Defaults to open, which is what a public demo
# backend wants; set CORS_ORIGINS to a comma-separated list to lock a deployment
# down to its own frontend.
_origins = os.getenv("CORS_ORIGINS", "*")
ALLOWED_ORIGINS = ["*"] if _origins.strip() == "*" else [
    o.strip() for o in _origins.split(",") if o.strip()]

CORS_CONFIG = {
    "allow_origins": ALLOWED_ORIGINS,
    # False on purpose. The API has no cookies or auth headers to carry, and the
    # combination of "allow any origin" with "send credentials" is one the CORS
    # spec forbids — browsers reject the wildcard outright once credentials are
    # in play, so asking for both invites a class of failure that appears only
    # in the deployed browser and never in local testing.
    "allow_credentials": False,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
