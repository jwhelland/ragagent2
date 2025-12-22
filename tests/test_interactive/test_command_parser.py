"""Tests for command parser."""

import pytest

from src.curation.interactive.command_parser import CommandHistory, CommandParser


class TestCommandParser:
    """Tests for CommandParser."""

    def test_parse_simple_command(self):
        """Test parsing a simple command without arguments."""
        parser = CommandParser()
        result = parser.parse("help")
        assert result.command == "help"
        assert result.args == []
        assert result.kwargs == {}
        assert result.raw_input == "help"

    def test_parse_command_with_args(self):
        """Test parsing a command with positional arguments."""
        parser = CommandParser()
        result = parser.parse("export results.json")
        assert result.command == "export"
        assert result.args == ["results.json"]
        assert result.kwargs == {}

    def test_parse_command_with_kwargs(self):
        """Test parsing a command with keyword arguments."""
        parser = CommandParser()
        result = parser.parse("filter type=SYSTEM status=pending")
        assert result.command == "filter"
        assert result.args == []
        assert result.kwargs == {"type": "SYSTEM", "status": "pending"}

    def test_parse_command_with_mixed_args(self):
        """Test parsing a command with both positional and keyword arguments."""
        parser = CommandParser()
        result = parser.parse("batch-approve >0.9 dry_run=true")
        assert result.command == "batch-approve"
        assert result.args == [">0.9"]
        assert result.kwargs == {"dry_run": "true"}

    def test_parse_command_with_leading_colon(self):
        """Test parsing a command with leading colon."""
        parser = CommandParser()
        result = parser.parse(":filter type=SYSTEM")
        assert result.command == "filter"
        assert result.kwargs == {"type": "SYSTEM"}

    def test_parse_empty_command(self):
        """Test parsing an empty command raises ValueError."""
        parser = CommandParser()
        with pytest.raises(ValueError, match="Empty command"):
            parser.parse("")

    def test_validate_filter_args_valid_type(self):
        """Test validating filter arguments with valid entity type."""
        parser = CommandParser()
        result = parser.validate_filter_args({"type": "SYSTEM"})
        assert result["entity_type"] == "SYSTEM"

    def test_validate_filter_args_invalid_type(self):
        """Test validating filter arguments with invalid entity type."""
        parser = CommandParser()
        with pytest.raises(ValueError, match="Invalid entity type"):
            parser.validate_filter_args({"type": "INVALID_TYPE"})

    def test_validate_filter_args_valid_status(self):
        """Test validating filter arguments with valid status."""
        parser = CommandParser()
        result = parser.validate_filter_args({"status": "pending"})
        assert result["status"] == "pending"

    def test_validate_filter_args_invalid_status(self):
        """Test validating filter arguments with invalid status."""
        parser = CommandParser()
        with pytest.raises(ValueError, match="Invalid status"):
            parser.validate_filter_args({"status": "invalid"})

    def test_validate_filter_args_valid_confidence(self):
        """Test validating filter arguments with valid confidence."""
        parser = CommandParser()
        result = parser.validate_filter_args({"confidence": "0.8"})
        assert result["min_confidence"] == 0.8

    def test_validate_filter_args_invalid_confidence_range(self):
        """Test validating filter arguments with confidence out of range."""
        parser = CommandParser()
        with pytest.raises(ValueError, match="Confidence must be between"):
            parser.validate_filter_args({"confidence": "1.5"})

    def test_parse_confidence_threshold_greater_than(self):
        """Test parsing confidence threshold with > operator."""
        parser = CommandParser()
        operator, value = parser.parse_confidence_threshold(">0.9")
        assert operator == ">"
        assert value == 0.9

    def test_parse_confidence_threshold_greater_equal(self):
        """Test parsing confidence threshold with >= operator."""
        parser = CommandParser()
        operator, value = parser.parse_confidence_threshold(">=0.8")
        assert operator == ">="
        assert value == 0.8

    def test_parse_confidence_threshold_less_than(self):
        """Test parsing confidence threshold with < operator."""
        parser = CommandParser()
        operator, value = parser.parse_confidence_threshold("<0.5")
        assert operator == "<"
        assert value == 0.5

    def test_parse_confidence_threshold_invalid_format(self):
        """Test parsing confidence threshold with invalid format."""
        parser = CommandParser()
        with pytest.raises(ValueError, match="Invalid threshold format"):
            parser.parse_confidence_threshold("0.9")

    def test_parse_confidence_threshold_out_of_range(self):
        """Test parsing confidence threshold with value out of range."""
        parser = CommandParser()
        with pytest.raises(ValueError, match="Threshold must be between"):
            parser.parse_confidence_threshold(">1.5")

    def test_validate_sort_args_valid_field(self):
        """Test validating sort arguments with valid field."""
        parser = CommandParser()
        result = parser.validate_sort_args(["confidence"])
        assert result == "confidence"

    def test_validate_sort_args_invalid_field(self):
        """Test validating sort arguments with invalid field."""
        parser = CommandParser()
        with pytest.raises(ValueError, match="Invalid sort field"):
            parser.validate_sort_args(["invalid_field"])

    def test_validate_sort_args_missing(self):
        """Test validating sort arguments with missing field."""
        parser = CommandParser()
        with pytest.raises(ValueError, match="Sort command requires a field name"):
            parser.validate_sort_args([])


class TestCommandHistory:
    """Tests for CommandHistory."""

    def test_add_command(self):
        """Test adding a command to history."""
        history = CommandHistory()
        history.add("filter type=SYSTEM")
        assert len(history.history) == 1
        assert history.history[0] == "filter type=SYSTEM"

    def test_add_duplicate_command(self):
        """Test that duplicate consecutive commands are not added."""
        history = CommandHistory()
        history.add("help")
        history.add("help")
        assert len(history.history) == 1

    def test_add_empty_command(self):
        """Test that empty commands are not added."""
        history = CommandHistory()
        history.add("")
        assert len(history.history) == 0

    def test_previous_command(self):
        """Test getting previous command from history."""
        history = CommandHistory()
        history.add("filter type=SYSTEM")
        history.add("batch-approve >0.9")

        # Navigate backwards
        result = history.previous()
        assert result == "batch-approve >0.9"

        result = history.previous()
        assert result == "filter type=SYSTEM"

        # At beginning, should stay at first command
        result = history.previous()
        assert result == "filter type=SYSTEM"

    def test_next_command(self):
        """Test getting next command from history."""
        history = CommandHistory()
        history.add("filter type=SYSTEM")
        history.add("batch-approve >0.9")

        # Navigate backwards first
        history.previous()
        history.previous()

        # Navigate forwards
        result = history.next()
        assert result == "batch-approve >0.9"

        # At end, should return empty string
        result = history.next()
        assert result == ""

    def test_reset_index(self):
        """Test resetting history navigation index."""
        history = CommandHistory()
        history.add("filter type=SYSTEM")
        history.add("batch-approve >0.9")

        # Navigate backwards
        history.previous()
        assert history.current_index == 1

        # Reset
        history.reset_index()
        assert history.current_index == 2

    def test_max_size_limit(self):
        """Test that history respects max size limit."""
        history = CommandHistory(max_size=3)
        history.add("command1")
        history.add("command2")
        history.add("command3")
        history.add("command4")

        # Should only keep last 3 commands
        assert len(history.history) == 3
        assert history.history[0] == "command2"
        assert history.history[1] == "command3"
        assert history.history[2] == "command4"
