class CollectionError(Exception):
    """Base exception for all collection-related errors in ZionDB."""
    pass


class CollectionAlreadyExistsError(CollectionError):
    """Raised when trying to create a collection that already exists."""
    pass


class CollectionNotFoundError(CollectionError):
    """Raised when trying to retrieve or delete a collection that does not exist."""
    pass
