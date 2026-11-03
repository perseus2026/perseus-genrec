"""Config-selected metric trackers."""

from genrecsys.core.registry import Registry
from genrecsys.core.trackers.base import Tracker
from genrecsys.core.trackers.clearml import ClearMLTracker
from genrecsys.core.trackers.logging import LoggingTracker

registry: Registry[type[Tracker]] = Registry("tracker")
registry.register("logging", LoggingTracker)
registry.register("clearml", ClearMLTracker)

__all__ = ["ClearMLTracker", "LoggingTracker", "Tracker", "registry"]
