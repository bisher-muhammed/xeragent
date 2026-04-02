#!/usr/bin/env python3
"""
Complete XER Data Extractor - Extracts ALL data from Primavera P6 XER files
No information is skipped - this is a comprehensive deep extraction system
"""

import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Any, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert any value to float, returning default on failure."""
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert any value to int, returning default on failure."""
    if value is None or value == '':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


class CompleteXERExtractor:
    """
    Comprehensive XER file parser that extracts EVERY piece of information
    including all tables, relationships, custom fields, and metadata.
    """

    def __init__(self, filepath: str, file_type: str = "baseline"):
        """
        Initialize the complete extractor.

        Args:
            filepath: Path to XER file
            file_type: "baseline" or "update"
        """
        self.filepath = Path(filepath)
        self.file_type = file_type
        self.filename = self.filepath.name

        # Core data structures - store EVERYTHING
        self.raw_content = ""
        self.metadata: Dict[str, Any] = {}
        self.tables: Dict[str, List[Dict]] = {}
        self.table_relationships: Dict[str, List[Dict]] = defaultdict(list)
        self.extraction_stats: Dict[str, Any] = {}
        self.parsing_errors: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all(self) -> 'CompleteXERExtractor':
        """Main extraction method - extracts EVERYTHING from the XER file."""
        print(f"Starting complete extraction of {self.filename}...")

        self._read_raw_content()
        self._parse_header()
        self._parse_all_tables()
        self._build_relationships()
        self._calculate_statistics()

        total_records = sum(len(t) for t in self.tables.values())
        print(
            f"Extraction complete: {len(self.tables)} tables, "
            f"{total_records} total records"
        )
        return self

    def get_complete_data(self) -> Dict[str, Any]:
        """
        Get ALL extracted data in a structured format.
        Returns EVERYTHING - no data is excluded.
        """
        return {
            'metadata': self.metadata,
            'tables': self.tables,
            'relationships': dict(self.table_relationships),
            'statistics': self.extraction_stats,
            'parsing_errors': self.parsing_errors,

            # Convenience accessors
            'project': self.get_project_info(),
            'tasks': self.get_all_tasks(),
            'resources': self.get_all_resources(),
            'calendars': self.get_all_calendars(),
            'wbs': self.get_wbs_structure(),
            'activity_codes': self.get_activity_codes(),
            'custom_fields': self.get_custom_fields(),
            'relationships_summary': self.get_relationships_summary(),
        }

    def get_project_info(self) -> Dict[str, Any]:
        """Extract comprehensive project information."""
        if 'PROJECT' not in self.tables or not self.tables['PROJECT']:
            return {}

        proj = self.tables['PROJECT'][0]

        # proj_short_name is the project ID/code; proj_name is the full name.
        # Many exports only populate proj_short_name, so fall back gracefully.
        project_name = (
            proj.get('proj_name', '')
            or proj.get('proj_short_name', '')
            or self.filename
        )
        project_code = proj.get('proj_short_name', '')

        return {
            'project_id': proj.get('proj_id', ''),
            'project_name': project_name,
            'project_code': project_code,
            'plan_start_date': proj.get('plan_start_date', ''),
            'plan_end_date': proj.get('plan_end_date', ''),
            'scheduled_end_date': proj.get('scd_end_date', ''),
            'data_date': proj.get('last_recalc_date', ''),
            'actual_start_date': proj.get('act_start_date', ''),
            'actual_end_date': proj.get('act_end_date', ''),
            'status_code': proj.get('status_code', ''),
            'critical_path_type': proj.get('critical_path_type', ''),
            'total_float_hours': _safe_float(proj.get('total_float_hr_cnt')),
            'orig_cost': _safe_float(proj.get('orig_cost')),
            'indep_remain_cost': _safe_float(proj.get('indep_remain_cost')),
            'project_flag': proj.get('project_flag', ''),
            'all_fields': proj,  # Always expose raw row
        }

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get ALL tasks with ALL fields."""
        return self.tables.get('TASK', [])

    def get_all_resources(self) -> List[Dict[str, Any]]:
        """Get ALL resources with ALL fields."""
        return self.tables.get('RSRC', [])

    def get_all_calendars(self) -> List[Dict[str, Any]]:
        """Get ALL calendars with ALL fields."""
        return self.tables.get('CALENDAR', [])

    def get_wbs_structure(self) -> List[Dict[str, Any]]:
        """Get complete WBS structure."""
        return self.tables.get('PROJWBS', [])

    def get_activity_codes(self) -> Dict[str, Any]:
        """Get all activity codes and their assignments."""
        return {
            'types': self.tables.get('ACTVTYPE', []),
            'values': self.tables.get('ACTVCODE', []),
            'assignments': self.tables.get('TASKACTV', []),
        }

    def get_custom_fields(self) -> Dict[str, Any]:
        """Get all custom field definitions and values."""
        return {
            'definitions': self.tables.get('UDFTYPE', []),
            'values': self.tables.get('UDFVALUE', []),
        }

    def get_relationships_summary(self) -> Dict[str, int]:
        """Get summary of all relationship counts."""
        return {
            'predecessor_links': len(self.table_relationships.get('TASK_PREDECESSORS', [])),
            'resource_assignments': len(self.table_relationships.get('TASK_RESOURCES', [])),
            'wbs_items': len(self.table_relationships.get('WBS_HIERARCHY', [])),
            'activity_code_assignments': len(self.table_relationships.get('TASK_ACTIVITY_CODES', [])),
            'role_assignments': len(self.table_relationships.get('TASK_ROLES', [])),
            'udf_values': len(self.table_relationships.get('UDF_VALUES', [])),
        }

    def save_to_json(self, output_path: str):
        """Save ALL extracted data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_complete_data(), f, indent=2, ensure_ascii=False)
        print(f"Complete data saved to: {output_path}")

    def generate_extraction_report(self) -> str:
        """Generate a human-readable report of what was extracted."""
        report_lines = [
            "=" * 80,
            "COMPLETE XER EXTRACTION REPORT",
            f"File: {self.filename}",
            f"Type: {self.file_type}",
            "=" * 80,
            "",
            "EXTRACTION STATISTICS:",
            f"  Total Tables Extracted : {self.extraction_stats.get('total_tables', 0)}",
            f"  Total Records Extracted: {self.extraction_stats.get('total_records', 0)}",
            f"  File Size              : {self.metadata.get('file_size_mb', 0)} MB",
            f"  Parsing Errors         : {len(self.parsing_errors)}",
            "",
            "TABLES EXTRACTED (with record counts):",
        ]

        for table_name in sorted(self.tables.keys()):
            count = len(self.tables[table_name])
            report_lines.append(f"  {table_name:30s} : {count:6d} records")

        summary = self.get_relationships_summary()
        report_lines.extend([
            "",
            "RELATIONSHIPS MAPPED:",
            f"  Predecessor Links        : {summary['predecessor_links']}",
            f"  Resource Assignments     : {summary['resource_assignments']}",
            f"  WBS Hierarchy Items      : {summary['wbs_items']}",
            f"  Activity Code Assignments: {summary['activity_code_assignments']}",
            f"  Role Assignments         : {summary['role_assignments']}",
            f"  UDF Values               : {summary['udf_values']}",
            "",
            "=" * 80,
        ])

        if self.parsing_errors:
            report_lines.append("PARSING ERRORS:")
            for err in self.parsing_errors[:20]:
                report_lines.append(f"  {err}")
            if len(self.parsing_errors) > 20:
                report_lines.append(f"  ... and {len(self.parsing_errors) - 20} more")

        return "\n".join(report_lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_raw_content(self):
        """Read the complete raw file content."""
        try:
            with open(self.filepath, 'r', encoding='windows-1252', errors='ignore') as f:
                self.raw_content = f.read()
        except Exception as exc:
            self.parsing_errors.append(f"Error reading file: {exc}")
            raise

    def _parse_header(self):
        """Parse XER file header with all metadata."""
        lines = self.raw_content.split('\n')
        if not lines:
            return

        parts = lines[0].strip().split('\t')
        if parts[0] != 'ERMHDR':
            self.parsing_errors.append("Header does not start with ERMHDR - file may be corrupt")
            return

        self.metadata = {
            'format': 'ERMHDR',
            'version': parts[1] if len(parts) > 1 else '',
            'export_date': parts[2] if len(parts) > 2 else '',
            'export_user_type': parts[3] if len(parts) > 3 else '',
            'export_username': parts[4] if len(parts) > 4 else '',
            'export_user_fullname': parts[5] if len(parts) > 5 else '',
            'database_name': parts[6] if len(parts) > 6 else '',
            'user_role': parts[7] if len(parts) > 7 else '',
            'currency_symbol': parts[8] if len(parts) > 8 else '',
            'file_type': self.file_type,
            'filename': self.filename,
            'file_size_bytes': len(self.raw_content),
            'file_size_mb': round(len(self.raw_content) / (1024 * 1024), 2),
        }

    def _parse_all_tables(self):
        """Parse ALL tables with ALL fields - nothing is skipped."""
        lines = self.raw_content.split('\n')

        current_table: Optional[str] = None
        current_fields: List[str] = []

        for line_num, line in enumerate(lines[1:], start=2):
            line = line.rstrip('\r')
            if not line.strip():
                continue

            parts = line.split('\t')
            marker = parts[0]

            try:
                if marker == '%T':
                    current_table = parts[1].strip() if len(parts) > 1 else ''
                    if current_table:
                        self.tables[current_table] = []
                    current_fields = []

                elif marker == '%F':
                    current_fields = [f.strip() for f in parts[1:]]

                elif marker == '%R':
                    if not current_table or not current_fields:
                        continue

                    row_data: Dict[str, Any] = {
                        '_table_name': current_table,
                        '_row_number': len(self.tables[current_table]) + 1,
                    }

                    # Capture ALL fields, including empty ones
                    for i, field in enumerate(current_fields):
                        raw_value = parts[i + 1] if i + 1 < len(parts) else ''
                        row_data[field] = raw_value

                    self.tables[current_table].append(row_data)

            except Exception as exc:
                self.parsing_errors.append(f"Line {line_num}: {exc}")

    def _build_relationships(self):
        """Build comprehensive relationship maps between all tables."""

        # Task predecessors / logic ties
        for pred in self.tables.get('TASKPRED', []):
            lag_hrs = _safe_float(pred.get('lag_hr_cnt', 0))
            self.table_relationships['TASK_PREDECESSORS'].append({
                'task_id': pred.get('task_id', ''),
                'predecessor_task_id': pred.get('pred_task_id', ''),
                'relationship_type': pred.get('pred_type', ''),
                'lag_hours': lag_hrs,
                'lag_days': lag_hrs / 8.0 if lag_hrs else 0.0,
            })

        # Resource assignments
        for rsrc in self.tables.get('TASKRSRC', []):
            self.table_relationships['TASK_RESOURCES'].append({
                'task_id': rsrc.get('task_id', ''),
                'resource_id': rsrc.get('rsrc_id', ''),
                'role_id': rsrc.get('role_id', ''),
                'budgeted_cost': _safe_float(rsrc.get('target_cost')),
                'actual_cost': _safe_float(rsrc.get('act_reg_cost')),
                'remaining_cost': _safe_float(rsrc.get('remain_cost')),
                'budgeted_qty': _safe_float(rsrc.get('target_qty')),
                'actual_qty': _safe_float(rsrc.get('act_reg_qty')),
                'remaining_qty': _safe_float(rsrc.get('remain_qty')),
            })

        # WBS hierarchy
        for wbs in self.tables.get('PROJWBS', []):
            self.table_relationships['WBS_HIERARCHY'].append({
                'wbs_id': wbs.get('wbs_id', ''),
                'parent_wbs_id': wbs.get('parent_wbs_id', ''),
                'wbs_name': wbs.get('wbs_name', ''),
                'wbs_short_name': wbs.get('wbs_short_name', ''),
                'wbs_level': _safe_int(wbs.get('obs_id', 0)),
            })

        # Activity code assignments
        for actv in self.tables.get('TASKACTV', []):
            self.table_relationships['TASK_ACTIVITY_CODES'].append({
                'task_id': actv.get('task_id', ''),
                'activity_code_id': actv.get('actv_code_id', ''),
                'activity_code_type_id': actv.get('actv_code_type_id', ''),
            })

        # Role assignments (inside TASKRSRC - records with a role_id)
        for rsrc in self.tables.get('TASKRSRC', []):
            role_id = rsrc.get('role_id', '')
            if role_id:
                self.table_relationships['TASK_ROLES'].append({
                    'task_id': rsrc.get('task_id', ''),
                    'role_id': role_id,
                    'resource_id': rsrc.get('rsrc_id', ''),
                })

        # UDF (User Defined Field) values
        for udf in self.tables.get('UDFVALUE', []):
            self.table_relationships['UDF_VALUES'].append({
                'fk_id': udf.get('fk_id', ''),
                'udf_type_id': udf.get('udf_type_id', ''),
                'udf_text': udf.get('udf_text', ''),
                'udf_number': _safe_float(udf.get('udf_number')),
                'udf_date': udf.get('udf_date', ''),
                'udf_code_id': udf.get('udf_code_id', ''),
            })

        # Notebook / text notes on tasks
        for nb in self.tables.get('TASKMEMO', []):
            self.table_relationships['TASK_NOTES'].append({
                'task_id': nb.get('task_id', ''),
                'memo_type_id': nb.get('memo_type_id', ''),
                'task_memo': nb.get('task_memo', ''),
            })

    def _calculate_statistics(self):
        """Calculate comprehensive statistics about the extraction."""
        total_records = sum(len(t) for t in self.tables.values())

        self.extraction_stats = {
            'total_tables': len(self.tables),
            'total_records': total_records,
            'file_type': self.file_type,
            'file_size_mb': self.metadata.get('file_size_mb', 0),
            'parsing_errors': len(self.parsing_errors),
            'extraction_timestamp': datetime.now().isoformat(),
        }

        # Per-table stats (exclude internal metadata fields from field count)
        self.extraction_stats['tables'] = {}
        for table_name, records in self.tables.items():
            if records:
                # Subtract the two internal metadata fields (_table_name, _row_number)
                field_count = max(0, len(records[0]) - 2)
                self.extraction_stats['tables'][table_name] = {
                    'record_count': len(records),
                    'field_count': field_count,
                }


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def extract_complete_xer_data(filepath: str, file_type: str = "baseline") -> CompleteXERExtractor:
    """
    Convenience function to extract all data from an XER file.

    Args:
        filepath: Path to XER file
        file_type: "baseline" or "update"

    Returns:
        CompleteXERExtractor with all data extracted
    """
    extractor = CompleteXERExtractor(filepath, file_type)
    extractor.extract_all()
    return extractor


if __name__ == "__main__":
    import sys

    search_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    xer_files = list(search_path.glob('*.xer'))

    if not xer_files:
        print("No XER files found!")
    else:
        for xer_file in xer_files:
            print(f"\n{'=' * 80}")
            print(f"Processing: {xer_file.name}")
            print('=' * 80)

            extractor = extract_complete_xer_data(str(xer_file), "baseline")
            print(extractor.generate_extraction_report())

            output_file = xer_file.stem + "_complete_data.json"
            extractor.save_to_json(output_file)

