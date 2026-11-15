"""Reverse proxy and buffering test runner alias for Ticket 10.

Imports all test suites from test_proxy_router to ensure backwards and
cross-reference compatibility across backlog specifications.
"""

from tests.test_proxy_router import *  # noqa: F401, F403
