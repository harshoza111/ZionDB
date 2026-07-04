import logging
from typing import Dict, Iterator, List

from ziondb.storage.storage import Storage
from ziondb.storage.record import ChunkRecord
from ziondb.storage.exceptions import RecordAlreadyExistsError, RecordNotFoundError

logger = logging.getLogger(__name__)

class InMemoryStorage(Storage):
    """An optimized in-memory implementation of the Storage interface."""

    def __init__(self) -> None:
        """Initialize empty records storage and index lookup dictionary."""
        self._records: List[ChunkRecord] = []
        # Maps record ID to its index in the _records list
        self._id_to_index: Dict[str, int] = {}

    def insert(self, record: ChunkRecord) -> None:
        """
        Inserts a new record.

        Args:
            record: The ChunkRecord to store.

        Raises:
            RecordAlreadyExistsError: If the record ID is already indexed.
        """
        if record.id in self._id_to_index:
            logger.warning(f"Failed to insert record: ID '{record.id}' already exists.")
            raise RecordAlreadyExistsError(f"Record with ID '{record.id}' already exists.")

        index = len(self._records)
        self._records.append(record)
        self._id_to_index[record.id] = index
        logger.debug(f"Inserted record '{record.id}' at index {index}.")

    def get(self, record_id: str) -> ChunkRecord:
        """
        Retrieves a record by its unique ID.

        Args:
            record_id: The ID of the record to fetch.

        Returns:
            ChunkRecord: The stored record.

        Raises:
            RecordNotFoundError: If the ID does not exist in storage.
        """
        if record_id not in self._id_to_index:
            raise RecordNotFoundError(f"Record with ID '{record_id}' not found.")

        index = self._id_to_index[record_id]
        return self._records[index]

    def update(self, record: ChunkRecord) -> None:
        """
        Updates an existing record in storage.

        Args:
            record: The updated ChunkRecord.

        Raises:
            RecordNotFoundError: If the record ID does not exist.
        """
        if record.id not in self._id_to_index:
            logger.warning(f"Failed to update record: ID '{record.id}' not found.")
            raise RecordNotFoundError(f"Record with ID '{record.id}' not found.")

        index = self._id_to_index[record.id]
        self._records[index] = record
        logger.debug(f"Updated record '{record.id}' at index {index}.")

    def delete(self, record_id: str) -> None:
        """
        Deletes a record from storage by its ID in O(1) time.
        Uses the swap-and-pop technique to avoid O(N) list-shift costs.

        Args:
            record_id: The ID of the record to delete.

        Raises:
            RecordNotFoundError: If the ID does not exist in storage.
        """
        if record_id not in self._id_to_index:
            logger.warning(f"Failed to delete record: ID '{record_id}' not found.")
            raise RecordNotFoundError(f"Record with ID '{record_id}' not found.")

        index_to_remove = self._id_to_index[record_id]
        last_index = len(self._records) - 1

        if index_to_remove == last_index:
            # It's the last element, simply pop it
            self._records.pop()
            del self._id_to_index[record_id]
        else:
            # Swap target element with the last element
            last_record = self._records[last_index]
            self._records[index_to_remove] = last_record
            self._id_to_index[last_record.id] = index_to_remove

            # Pop the target element now at the end of the list
            self._records.pop()
            del self._id_to_index[record_id]

        logger.debug(f"Deleted record '{record_id}' from index {index_to_remove}.")

    def exists(self, record_id: str) -> bool:
        """
        Checks if a record with the specified ID exists in storage.

        Args:
            record_id: The ID to check.

        Returns:
            bool: True if it exists, False otherwise.
        """
        return record_id in self._id_to_index

    def iterate(self) -> Iterator[ChunkRecord]:
        """
        Returns an iterator over all stored records.

        Returns:
            Iterator[ChunkRecord]: An iterator yielding ChunkRecords.
        """
        return iter(self._records)

    def clear(self) -> None:
        """Removes all records and indexes from storage."""
        self._records.clear()
        self._id_to_index.clear()
        logger.debug("Cleared all records from storage.")

    def size(self) -> int:
        """
        Returns the total number of records currently stored.

        Returns:
            int: The record count.
        """
        return len(self._records)
