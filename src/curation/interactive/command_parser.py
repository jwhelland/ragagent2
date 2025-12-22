"""Command parser for vim-style command mode."""

import re
from dataclasses import dataclass
from typing import Any, Optional

from src.storage.schemas import EntityType


@dataclass
class ParsedCommand:
    """Result of parsing a command."""

    command: str  # The base command (e.g., "filter", "sort", "batch-approve")
    args: list[str]  # Positional arguments
    kwargs: dict[str, str]  # Named arguments (key=value pairs)
    raw_input: str  # Original command string


class CommandParser:
    """Parser for vim-style commands in the interactive review interface."""

    # Regex patterns for parsing
    KEY_VALUE_PATTERN = re.compile(r"(\w+)=([^\s]+)")
    OPERATOR_VALUE_PATTERN = re.compile(r"([><=]+)([\d.]+)")

    def parse(self, command_str: str) -> ParsedCommand:
        """Parse a command string into structured components.

        Args:
            command_str: The command string (without leading ':')

        Returns:
            ParsedCommand with parsed components

        Examples:
            >>> parser.parse("filter type=SYSTEM status=pending")
            ParsedCommand(command='filter', args=[], kwargs={'type': 'SYSTEM', 'status': 'pending'}, ...)

            >>> parser.parse("batch-approve >0.9")
            ParsedCommand(command='batch-approve', args=['>0.9'], kwargs={}, ...)

            >>> parser.parse("export results.json")
            ParsedCommand(command='export', args=['results.json'], kwargs={}, ...)
        """
        # Strip whitespace and remove leading ':' if present
        command_str = command_str.strip().lstrip(":")

        if not command_str:
            raise ValueError("Empty command")

        # Split into tokens
        tokens = command_str.split()
        if not tokens:
            raise ValueError("Empty command")

        # First token is the command
        command = tokens[0].lower()
        remaining_tokens = tokens[1:]

        # Parse arguments and kwargs
        args = []
        kwargs = {}

        for token in remaining_tokens:
            # Check for key=value pattern
            kv_match = self.KEY_VALUE_PATTERN.match(token)
            if kv_match:
                key, value = kv_match.groups()
                kwargs[key.lower()] = value
            else:
                # It's a positional argument
                args.append(token)

        return ParsedCommand(
            command=command,
            args=args,
            kwargs=kwargs,
            raw_input=command_str,
        )

    def validate_filter_args(self, kwargs: dict[str, str]) -> dict[str, Any]:
        """Validate and normalize filter command arguments.

        Args:
            kwargs: Parsed keyword arguments

        Returns:
            Dictionary of validated filter parameters

        Raises:
            ValueError: If validation fails
        """
        result = {}

        # Validate type
        if "type" in kwargs:
            type_value = kwargs["type"].upper()
            try:
                result["entity_type"] = EntityType[type_value].value
            except KeyError:
                valid_types = ", ".join(t.name for t in EntityType)
                raise ValueError(f"Invalid entity type '{type_value}'. Valid: {valid_types}")

        # Validate status
        if "status" in kwargs:
            status = kwargs["status"].lower()
            if status not in ("pending", "approved", "rejected", "all"):
                raise ValueError(
                    f"Invalid status '{status}'. Valid: pending, approved, rejected, all"
                )
            result["status"] = status

        # Validate confidence
        if "confidence" in kwargs or "min_confidence" in kwargs:
            confidence_str = kwargs.get("confidence") or kwargs.get("min_confidence")
            try:
                confidence = float(confidence_str)
                if not (0.0 <= confidence <= 1.0):
                    raise ValueError("Confidence must be between 0.0 and 1.0")
                result["min_confidence"] = confidence
            except ValueError as e:
                raise ValueError(f"Invalid confidence value '{confidence_str}': {e}")

        return result

    def parse_confidence_threshold(self, arg: str) -> tuple[str, float]:
        """Parse a confidence threshold argument like '>0.9', '>=0.8', etc.

        Args:
            arg: Argument string with operator and value

        Returns:
            Tuple of (operator, threshold_value)

        Raises:
            ValueError: If format is invalid
        """
        match = self.OPERATOR_VALUE_PATTERN.match(arg)
        if not match:
            raise ValueError(
                f"Invalid threshold format '{arg}'. Expected format: >0.9, >=0.8, etc."
            )

        operator, value_str = match.groups()
        try:
            value = float(value_str)
            if not (0.0 <= value <= 1.0):
                raise ValueError("Threshold must be between 0.0 and 1.0")
        except ValueError as e:
            raise ValueError(f"Invalid threshold value '{value_str}': {e}")

        return operator, value

    def validate_sort_args(self, args: list[str]) -> str:
        """Validate sort command arguments.

        Args:
            args: Positional arguments

        Returns:
            Sort field name

        Raises:
            ValueError: If validation fails
        """
        if not args:
            raise ValueError("Sort command requires a field name (e.g., 'confidence', 'name')")

        field = args[0].lower()
        valid_fields = ["confidence", "name", "type", "mentions"]

        if field not in valid_fields:
            raise ValueError(f"Invalid sort field '{field}'. Valid: {', '.join(valid_fields)}")

        return field


class CommandHistory:
    """Manages command history for command mode."""

    def __init__(self, max_size: int = 100):
        """Initialize command history.

        Args:
            max_size: Maximum number of commands to keep in history
        """
        self.max_size = max_size
        self.history: list[str] = []
        self.current_index: int = -1

    def add(self, command: str) -> None:
        """Add a command to history.

        Args:
            command: Command string to add
        """
        # Don't add empty commands or duplicates of the last command
        if command and (not self.history or self.history[-1] != command):
            self.history.append(command)
            # Trim history if needed
            if len(self.history) > self.max_size:
                self.history = self.history[-self.max_size :]
        # Reset index
        self.current_index = len(self.history)

    def previous(self) -> Optional[str]:
        """Get the previous command in history.

        Returns:
            Previous command or None if at beginning
        """
        if not self.history:
            return None

        if self.current_index > 0:
            self.current_index -= 1
            return self.history[self.current_index]
        elif self.current_index == 0:
            return self.history[0]
        else:
            self.current_index = 0
            return self.history[0]

    def next(self) -> Optional[str]:
        """Get the next command in history.

        Returns:
            Next command or empty string if at end
        """
        if not self.history:
            return ""

        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            return self.history[self.current_index]
        else:
            self.current_index = len(self.history)
            return ""

    def reset_index(self) -> None:
        """Reset history navigation index to end."""
        self.current_index = len(self.history)
