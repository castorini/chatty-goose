import os

import coloredlogs


coloredlogs.install(
    level=os.environ.get("CHATTY_GOOSE_LOG_LEVEL", "WARN"),
    fmt="%(asctime)s [%(levelname)s]: %(message)s",
)
