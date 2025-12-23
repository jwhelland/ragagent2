"""Textual widgets for the interactive review interface."""

from src.curation.interactive.widgets.batch_modal import BatchOperationModal
from src.curation.interactive.widgets.candidate_list import CandidateList, CandidateRow
from src.curation.interactive.widgets.command_modal import CommandModalScreen
from src.curation.interactive.widgets.comparison_modal import ComparisonModalScreen
from src.curation.interactive.widgets.detail_panel import DetailPanel, DetailSection
from src.curation.interactive.widgets.duplicate_suggestions import DuplicateSuggestionsPanel
from src.curation.interactive.widgets.edit_modal import EditModalScreen
from src.curation.interactive.widgets.entity_candidate_merge_preview_modal import (
    EntityCandidateMergePreviewModal,
)
from src.curation.interactive.widgets.entity_search_modal import EntitySearchModal
from src.curation.interactive.widgets.merge_preview_modal import MergePreviewModal
from src.curation.interactive.widgets.primary_selection_modal import PrimarySelectionModal
from src.curation.interactive.widgets.search_modal import SearchFilters, SearchModalScreen
from src.curation.interactive.widgets.status_bar import STATUS_BAR_CSS, StatusBar

__all__ = [
    "BatchOperationModal",
    "CandidateList",
    "CandidateRow",
    "CommandModalScreen",
    "ComparisonModalScreen",
    "DetailPanel",
    "DetailSection",
    "DuplicateSuggestionsPanel",
    "EditModalScreen",
    "EntitySearchModal",
    "EntityCandidateMergePreviewModal",
    "MergePreviewModal",
    "PrimarySelectionModal",
    "SearchFilters",
    "SearchModalScreen",
    "STATUS_BAR_CSS",
    "StatusBar",
]
