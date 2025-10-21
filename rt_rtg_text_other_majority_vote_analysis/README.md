# Majority Vote Analysis of Text and Other resourceType/resourceTypeGeneral Pairings

Tool for majority vote analysis of `resourceType` - `resourceTypeGeneral` pairings in DataCite for works where resourceTypeGeneral is `Text` or `Other`. 


## Prerequisites

- Parse all resourceType and resourceTypeGeneral values to a CSV file with [fast-field-parser](https://github.com/adambuttrick/datacite-utils/tree/main/fast-field-parser)

## Build the Rust binary

```bash
cargo build --release
```

Which generates the release binary `target/release/rt_rtg_majority_vote_analysis`.

## Running the analysis


```bash
./target/release/rt_rtg_majority_vote_analysis \
  -i rt_rtg_filtered.csv \
  -o majority_vote_analysis.csv
```

Behaviour:
- Processes all DOIs with `resourceTypeGeneral` `Text` or `Other`.
- Suggests candidate reclassifcations for both groups, excluding recommendations to map Text to Text or Text to Other.
- Blocks Other to Other and Other to Text reassignments, keeps only genuinely new resourceTypeGeneral values.

### Allow “Other” to become “Text”

If you want to consider `Other` records for reassignment to `Text`, use:

```bash
./target/release/rt_rtg_majority_vote_analysis \
  -i rt_rtg_filtered.csv \
  -o majority_vote_analysis.csv \
  --allow-other-to-text
```

### Tuning the heuristics

Key flags:

- `--min-count <int>` (default `5`): minimum number of DOIs sharing a `resourceType` under the same general before analysis.
- `--match-threshold <float>` (default `0.86`): normalized Levenshtein cutoff for treating two `resourceType` strings as variants.
- `--majority-threshold <float>` (default `0.55`): minimum vote share the leading candidate general must secure.
- `--max-candidates <int>` (default `3`): number of ranked alternatives written to the CSV.
- `--log-level <level>`: choose `debug`, `info`, `warn`, or `error` (default `info`).

Example: stricter similarity and higher support

```bash
./target/release/rt_rtg_majority_vote_analysis \
  -i rt_rtg_filtered.csv \
  -o majority_vote_analysis.csv \
  --min-count 10 \
  --match-threshold 0.9 \
  --majority-threshold 0.65
```

## Output format

`majority_vote_analysis.csv` includes:

- `normalized_resource_type`: canonicalised key used for clustering.
- `representative_resource_type`: most common raw `resourceType` among the source records.
- `source_resourceTypeGeneral` / `source_doi_count`: which group (`Text` or `Other`) generated the suggestion and how many DOIs it covers.
- `majority_resourceTypeGeneral` / `majority_ratio`: top candidate general and its share of the vote.
- `candidate_total`: total votes accumulated across non-excluded general types.
- `matched_variant_count`: number of fuzzy-matched `resourceType` variants contributing to the vote.
- `matched_examples`: sample variant strings with similarity scores.
- `source_examples`: top raw `resourceType` values from the original group.
- `candidate_{n}_*`: ranked alternative `resourceTypeGeneral` suggestions with counts and ratios.

Rows are sorted by descending impact (`source_doi_count`) and confidence (`majority_ratio`).
