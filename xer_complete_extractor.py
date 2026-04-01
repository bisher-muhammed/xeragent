#!/usr/bin/env python3
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Any, Optional


class CompleteXERExtractor:

    def __init__(self, filepath: str, file_type: str = "baseline"):
        self.filepath = Path(filepath)
        self.file_type = file_type
        self.filename = self.filepath.name
        self.raw_content = ""
        self.metadata = {}
        self.tables = {}
        self.table_relationships = defaultdict(list)
        self.extraction_stats = {}
        self.parsing_errors = []
        self._primary_keys = {}
        self.orphan_references = []

    def extract_all(self) -> 'CompleteXERExtractor':
        self._read_raw_content()
        self._parse_header()
        self._parse_all_tables()
        self._build_relationships()
        self._validate_relationships()
        self._calculate_statistics()
        print(f"Extraction complete: {len(self.tables)} tables, "
              f"{sum(len(t) for t in self.tables.values())} total records")
        return self

    def _read_raw_content(self):
        try:
            with open(self.filepath, 'r', encoding='windows-1252', errors='ignore') as f:
                self.raw_content = f.read()
        except Exception as e:
            self.parsing_errors.append(f"Error reading file: {str(e)}")
            raise

    def _parse_header(self):
        lines = self.raw_content.split('\n')
        if not lines:
            return
        parts = lines[0].strip().split('\t')
        if parts[0] == 'ERMHDR':
            self.metadata = {
                'format': 'ERMHDR',
                'version': parts[1] if len(parts) > 1 else '',
                'export_date': parts[2] if len(parts) > 2 else '',
                'export_username': parts[4] if len(parts) > 4 else '',
                'export_user_fullname': parts[5] if len(parts) > 5 else '',
                'database_name': parts[6] if len(parts) > 6 else '',
                'currency_symbol': parts[8] if len(parts) > 8 else '',
                'file_type': self.file_type,
                'filename': self.filename,
                'file_size_bytes': len(self.raw_content),
                'file_size_mb': round(len(self.raw_content) / (1024 * 1024), 2)
            }

    def _parse_all_tables(self, error_threshold: float = 0.05) -> None:
        lines = self.raw_content.split('\n')
        current_table = None
        current_fields = []
        table_errors = defaultdict(int)

        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')

            try:
                if parts[0] == '%T':
                    current_table = parts[1] if len(parts) > 1 else ''
                    self.tables[current_table] = []
                    current_fields = []

                elif parts[0] == '%F':
                    current_fields = parts[1:]

                elif parts[0] == '%R' and current_table and current_fields:
                    row_data = {}
                    for i, field in enumerate(current_fields):
                        raw_value = parts[i + 1] if i + 1 < len(parts) else ""
                        try:
                            row_data[field] = self._infer_and_cast(field, raw_value)
                        except Exception as e:
                            table_errors[current_table] += 1
                            self.parsing_errors.append(
                                f"Line {line_num}, {current_table}.{field}: {str(e)}"
                            )
                            row_data[field] = None
                    self.tables[current_table].append(row_data)

            except Exception as e:
                table_errors[current_table] += 1
                self.parsing_errors.append(f"Line {line_num}, Table {current_table}: {str(e)}")

        for table, err_count in table_errors.items():
            total = len(self.tables.get(table, []))
            if total and (err_count / total) > error_threshold:
                print(f"WARNING: {table} has {err_count}/{total} parsing errors "
                      f"({err_count / total * 100:.1f}%)")

    def _infer_and_cast(self, field: str, raw_value: str) -> Any:
        val = raw_value.strip()
        if val == "" or val.upper() in {"NULL", "NONE"}:
            return None

        lower = field.lower()

        # Boolean: XER uses Y/N for flag fields
        if lower.startswith("is_") or lower.endswith("_flag"):
            if val.upper() in {"Y", "YES", "TRUE", "1"}:
                return True
            if val.upper() in {"N", "NO", "FALSE", "0"}:
                return False

        # Dates: XER uses "YYYY-MM-DD HH:MM" format primarily
        if "date" in lower:
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%d-%b-%y", "%m/%d/%Y"):
                try:
                    return datetime.strptime(val, fmt)
                except Exception:
                    continue
            return val

        # Numeric fields
        numeric_keywords = ("cnt", "cost", "qty", "hr_cnt", "duration", "pct", "float", "wt", "seq", "lag")
        if any(kw in lower for kw in numeric_keywords):
            try:
                return int(val)
            except Exception:
                try:
                    return float(val)
                except Exception:
                    return val

        # Numeric IDs
        if lower.endswith("_id") and val.isdigit():
            return int(val)

        return val

    def _build_relationships(self) -> None:
        # Identify primary key for each table (first _id field)
        for table_name, rows in self.tables.items():
            if not rows:
                continue
            for col in rows[0].keys():
                if col.lower().endswith("_id"):
                    self._primary_keys[table_name] = col
                    break

        # Check for duplicate PKs
        duplicates = []
        for tbl, pk in self._primary_keys.items():
            seen = set()
            for r in self.tables.get(tbl, []):
                val = r.get(pk)
                if val in seen:
                    duplicates.append({"table": tbl, "pk": pk, "value": val})
                else:
                    seen.add(val)

        self.extraction_stats['duplicate_pk_count'] = len(duplicates)
        self.extraction_stats['duplicate_pk_records'] = duplicates[:50]  # cap stored records

        pk_to_table = {pk: tbl for tbl, pk in self._primary_keys.items()}

        # Schema-level FK relationships — one edge per (from_table, to_table, via_field), not per row
        for table_name, rows in self.tables.items():
            if not rows:
                continue
            own_pk = self._primary_keys.get(table_name)
            seen_edges = set()
            for field_name in rows[0].keys():
                if field_name == own_pk:
                    continue
                ref_table = pk_to_table.get(field_name)
                if ref_table and ref_table != table_name:
                    edge_key = (ref_table, field_name)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        self.table_relationships[table_name].append({
                            "from_table": table_name,
                            "to_table": ref_table,
                            "via_field": field_name
                        })

    def _validate_relationships(self) -> None:
        pk_values: Dict[str, set] = {
            tbl: {row.get(pk) for row in self.tables.get(tbl, [])}
            for tbl, pk in self._primary_keys.items()
        }

        orphan_count = 0
        orphan_samples = []

        for from_tbl, edges in self.table_relationships.items():
            for edge in edges:
                to_tbl = edge["to_table"]
                via_field = edge["via_field"]
                if to_tbl not in pk_values:
                    continue
                valid_pks = pk_values[to_tbl]
                for row in self.tables.get(from_tbl, []):
                    val = row.get(via_field)
                    if val is None or val == "":
                        continue
                    if val not in valid_pks:
                        orphan_count += 1
                        if len(orphan_samples) < 100:
                            orphan_samples.append({
                                "from_table": from_tbl,
                                "to_table": to_tbl,
                                "via_field": via_field,
                                "value": val
                            })

        self.orphan_references = orphan_samples
        self.extraction_stats["orphan_fk_count"] = orphan_count

    def _calculate_statistics(self):
        # Preserve stats already computed by earlier methods
        preserved = {
            'duplicate_pk_count': self.extraction_stats.get('duplicate_pk_count', 0),
            'duplicate_pk_records': self.extraction_stats.get('duplicate_pk_records', []),
            'orphan_fk_count': self.extraction_stats.get('orphan_fk_count', 0),
        }

        self.extraction_stats = {
            **preserved,
            'total_tables': len(self.tables),
            'total_records': sum(len(t) for t in self.tables.values()),
            'file_type': self.file_type,
            'file_size_mb': self.metadata.get('file_size_mb', 0),
            'parsing_errors': len(self.parsing_errors),
            'extraction_timestamp': datetime.now().isoformat(),
            'tables': {
                name: {
                    'record_count': len(records),
                    'field_count': len(records[0]) if records else 0
                }
                for name, records in self.tables.items()
            }
        }

    def get_complete_data(self) -> Dict[str, Any]:
        return {
            'metadata': self.metadata,
            'tables': self.tables,
            'relationships': dict(self.table_relationships),
            'orphan_references': self.orphan_references,
            'statistics': self.extraction_stats,
            'parsing_errors': self.parsing_errors,
            'project': self.get_project_info(),
            'tasks': self.get_all_tasks(),
            'resources': self.get_all_resources(),
            'calendars': self.get_all_calendars(),
            'wbs': self.get_wbs_structure(),
            'activity_codes': self.get_activity_codes(),
            'custom_fields': self.get_custom_fields(),
            'relationships_summary': self.get_relationships_summary()
        }

    def get_project_info(self) -> Dict[str, Any]:
        if 'PROJECT' not in self.tables or not self.tables['PROJECT']:
            return {}
        proj = self.tables['PROJECT'][0]
        return {
            'project_id': proj.get('proj_id', ''),
            'project_name': proj.get('proj_short_name', ''),
            'full_name': proj.get('proj_short_name', ''),
            'plan_start_date': proj.get('plan_start_date', ''),
            'plan_end_date': proj.get('plan_end_date', ''),
            'scheduled_end_date': proj.get('scd_end_date', ''),
            'data_date': proj.get('last_recalc_date', ''),
            'actual_start_date': proj.get('act_start_date', ''),
            'actual_end_date': proj.get('act_end_date', ''),
            'status_code': proj.get('status_code', ''),
            'critical_path_type': proj.get('critical_path_type', ''),
            'total_float_hours': proj.get('total_float_hr_cnt', ''),
            'orig_cost': proj.get('orig_cost', ''),
            'indep_remain_cost': proj.get('indep_remain_cost', ''),
            'all_fields': proj
        }

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        return self.tables.get('TASK', [])

    def get_all_resources(self) -> List[Dict[str, Any]]:
        return self.tables.get('RSRC', [])

    def get_all_calendars(self) -> List[Dict[str, Any]]:
        return self.tables.get('CALENDAR', [])

    def get_wbs_structure(self) -> List[Dict[str, Any]]:
        return self.tables.get('PROJWBS', [])

    def get_activity_codes(self) -> Dict[str, Any]:
        return {
            'types': self.tables.get('ACTVTYPE', []),
            'values': self.tables.get('ACTVCODE', []),
            'assignments': self.tables.get('TASKACTV', [])
        }

    def get_custom_fields(self) -> Dict[str, Any]:
        return {
            'definitions': self.tables.get('UDFTYPE', []),
            'values': self.tables.get('UDFVALUE', [])
        }

    def get_relationships_summary(self) -> Dict[str, int]:
        return {tbl: len(edges) for tbl, edges in self.table_relationships.items()}

    def save_to_json(self, output_path: str):
        def _serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_complete_data(), f, indent=2, ensure_ascii=False, default=_serialize)
        print(f"Data saved to: {output_path}")

    def generate_extraction_report(self) -> str:
        lines = [
            "=" * 80,
            "XER EXTRACTION REPORT",
            f"File: {self.filename} ({self.file_type})",
            "=" * 80,
            f"Total Tables:    {self.extraction_stats.get('total_tables', 0)}",
            f"Total Records:   {self.extraction_stats.get('total_records', 0)}",
            f"File Size:       {self.metadata.get('file_size_mb', 0)} MB",
            f"Parsing Errors:  {len(self.parsing_errors)}",
            f"Duplicate PKs:   {self.extraction_stats.get('duplicate_pk_count', 0)}",
            f"Orphan FKs:      {self.extraction_stats.get('orphan_fk_count', 0)}",
            "",
            "TABLES:"
        ]
        for name in sorted(self.tables.keys()):
            lines.append(f"  {name:30s}: {len(self.tables[name]):6d} records")
        return "\n".join(lines)


def extract_complete_xer_data(filepath: str, file_type: str = "baseline") -> CompleteXERExtractor:
    extractor = CompleteXERExtractor(filepath, file_type)
    extractor.extract_all()
    return extractor


if __name__ == "__main__":
    xer_files = list(Path('.').glob('*.xer'))
    if not xer_files:
        print("No XER files found!")
    else:
        for xer_file in xer_files:
            extractor = extract_complete_xer_data(str(xer_file))
            print(extractor.generate_extraction_report())
            extractor.save_to_json(xer_file.stem + "_data.json")
