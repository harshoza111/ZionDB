class StorageError(Exception):
    """Base exception class for all storage operations in ZionDB."""
    pass

class RecordNotFoundError(StorageError):
    """Raised when a record cannot be found in storage by its ID."""
    pass

class RecordAlreadyExistsError(StorageError):
    """Raised when trying to insert a record with an ID that already exists in storage."""
    pass
