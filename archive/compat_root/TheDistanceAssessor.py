"""레거시 루트 진입점 호환 wrapper."""

from _root_wrapper import expose_or_run

expose_or_run("archive.legacy.TheDistanceAssessor", globals(), __name__)

