"""Content-addressed, read-only snapshots independent of mutable task files."""

import errno
import hashlib
import os
import tempfile
from pathlib import Path

from reward_engine.adapters.integrity_cache import identity


class EvidenceSnapshots:
    def __init__(self, directory, cache):
        self.directory = Path(directory)
        self.cache = cache

    def copy(self, source, destination):
        source, destination = Path(source), Path(destination)
        before = identity(source)
        digest = self.cache.digest(source)
        target = self.directory / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            descriptor, temporary = tempfile.mkstemp(prefix='.snapshot-', dir=target.parent)
            temporary = Path(temporary)
            try:
                copied = hashlib.sha256()
                with source.open('rb') as incoming, os.fdopen(descriptor, 'wb') as outgoing:
                    for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                        outgoing.write(chunk)
                        copied.update(chunk)
                if copied.hexdigest() != digest or before != identity(source):
                    raise ValueError('Evidence source changed while creating snapshot')
                temporary.chmod(0o444)
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
        if self.cache.digest(target) != digest:
            raise ValueError('Archived task evidence object changed')
        if identity(target)[2] & 0o222:
            raise ValueError('Evidence snapshot object must remain read-only')
        if before != identity(source):
            raise ValueError('Evidence source changed while creating snapshot')
        # Only immutable archive objects are linked. Mutable task files never are.
        try:
            os.link(target, destination, follow_symlinks=False)
        except OSError as error:
            if error.errno != errno.EMLINK:
                raise
            self._copy_without_link(target, destination, digest)
        return digest

    @staticmethod
    def _copy_without_link(target, destination, digest):
        """Retain saturated objects and publish a separate verified read-only copy."""
        created = False
        try:
            copied = hashlib.sha256()
            with destination.open('xb') as outgoing:
                created = True
                with target.open('rb') as incoming:
                    for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                        outgoing.write(chunk)
                        copied.update(chunk)
            if copied.hexdigest() != digest:
                raise ValueError('Evidence snapshot object changed during copy')
            destination.chmod(0o444)
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise
