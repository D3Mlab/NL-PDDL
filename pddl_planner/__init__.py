import logging
import sys

# Package-level loggers.  Users control verbosity via standard logging config
# or the convenience ``verbose`` / ``llm_verbose`` flags on NLFOLRegressionPlanner.
#
# Logger hierarchy:
#   pddl_planner            – package root (rarely used directly)
#   pddl_planner.planner    – regression planner: depth, frontier, SSA, entailment routing
#   pddl_planner.llm        – LLM calls: prompts, cache, retries, responses
#   pddl_planner.domain     – domain/instance parsing

logging.getLogger(__name__).addHandler(logging.NullHandler())


class ColoredFormatter(logging.Formatter):
    """Logging formatter that adds ANSI color codes based on log level.

    Falls back to plain text when the stream is not a terminal.
    """

    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG:    "\033[36m",    # cyan
        logging.INFO:     "\033[32m",    # green
        logging.WARNING:  "\033[33m",    # yellow
        logging.ERROR:    "\033[31m",    # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    NAME_COLORS = {
        "pddl_planner.planner": "\033[35m",  # magenta
        "pddl_planner.llm":     "\033[34m",  # blue
        "pddl_planner.domain":  "\033[36m",  # cyan
        "pddl_planner.cli":     "\033[37m",  # white
    }

    def __init__(self, fmt: str | None = None, use_color: bool = True):
        super().__init__(fmt or "%(name)s | %(levelname)s | %(message)s")
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_color:
            return super().format(record)

        level_color = self.COLORS.get(record.levelno, "")
        name_color = self.NAME_COLORS.get(record.name, "\033[90m")  # grey default
        reset = self.RESET

        orig_name = record.name
        orig_levelname = record.levelname
        orig_msg = record.msg

        record.name = f"{name_color}{record.name}{reset}"
        record.levelname = f"{level_color}{record.levelname}{reset}"
        record.msg = f"{level_color}{record.msg}{reset}"

        result = super().format(record)

        record.name = orig_name
        record.levelname = orig_levelname
        record.msg = orig_msg
        return result


def make_colored_handler(stream=None, use_color: bool | None = None) -> logging.StreamHandler:
    """Create a StreamHandler with ColoredFormatter.

    ``use_color`` defaults to True when the stream is a TTY.
    """
    stream = stream or sys.stdout
    if use_color is None:
        use_color = hasattr(stream, "isatty") and stream.isatty()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ColoredFormatter(use_color=use_color))
    return handler
