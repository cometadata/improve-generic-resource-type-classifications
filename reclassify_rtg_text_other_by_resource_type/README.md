# Reclassify resourceTypeGeneral Text/Other by resourceType

Reclassifies `resourceTypeGeneral` values in DataCite DOIs by matching their `resourceType` values against reference `resourceTypeGeneral` values using fuzzy matching.

## Installation
```bash
pip install pyyaml python-Levenshtein tqdm
```

## Usage

```bash
python reclassify_rtg_text_other_by_resource_type -i input.csv -r reference.yaml -o output.csv
```

## Arguments

- `-i, --input`: Input CSV file (required)
- `-o, --output`: Output CSV file (required)
- `-r, --reference`: Reference YAML file (default: resource_type_general_values.yaml)
- `-t, --threshold`: Fuzzy match threshold 0-1 (default: 0.95)
- `-c, --chunksize`: Rows per batch (default: 100000)
- `-v, --verbose`: Enable verbose output
- `--log-file`: Custom log file path
- `--log-level`: Log level (DEBUG/INFO/WARNING/ERROR)

## Input Format

CSV with columns:
- `doi`: DOI identifier
- `subfield_path`: Field path (types.resourceType or types.resourceTypeGeneral)
- `value`: Field value

The input CSV can be derived from the DataCite data file using the [fast-field-parser](https://github.com/adambuttrick/datacite-utils/tree/main/fast-field-parser) utility.

## Output Format

CSV with columns:
- `doi`: DOI identifier
- `resourceType`: Original resourceType value
- `resourceTypeGeneral`: Original resourceTypeGeneral value
- `inferredResourceTypeGeneral`: Matched reference value