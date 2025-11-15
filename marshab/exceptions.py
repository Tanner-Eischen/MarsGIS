"""Custom exceptions for MarsHab."""


class MarsHabError(Exception):
    """Base exception for all MarsHab errors."""

    def __init__(self, message: str, details: dict | None = None):
        """Initialize exception.

        Args:
            message: Error message
            details: Optional dictionary with additional context
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class DataError(MarsHabError):
    """Data acquisition, validation, or I/O errors."""
    pass


class AnalysisError(MarsHabError):
    """Analysis pipeline errors."""
    pass


class CoordinateError(MarsHabError):
    """Coordinate transformation errors."""
    pass


class NavigationError(MarsHabError):
    """Path planning and navigation errors."""
    pass


class ConfigurationError(MarsHabError):
    """Configuration loading or validation errors."""
    pass

