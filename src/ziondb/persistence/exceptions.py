class PersistenceError(Exception):
    """Base exception for all persistence-related errors in ZionDB."""
    pass


class SerializationError(PersistenceError):
    """Raised when serialization or deserialization fails."""
    pass


class InvalidPersistenceVersion(PersistenceError):
    """Raised when loading a collection with an unsupported persistence format version."""
    pass


class CorruptedCollectionError(PersistenceError):
    """Raised when reading files that have corrupted binary structures or mismatching headers."""
    pass
