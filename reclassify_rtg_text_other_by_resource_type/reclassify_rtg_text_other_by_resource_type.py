import csv
import re
import sys
import logging
import argparse
import yaml
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import Levenshtein
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classify DOIs based on resourceType matching against reference values'
    )
    parser.add_argument('-i', '--input', required=True,
                        type=str, help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True,
                        type=str, help='Output CSV file path')
    parser.add_argument('-r', '--reference', default='resource_type_general_values.yaml',
                        type=str, help='Reference values file path (default: resource_type_general_values.yaml)')
    parser.add_argument('-c', '--chunksize', default=100000, type=int,
                        help='Number of rows to process at once (default: 100000)')
    parser.add_argument('-t', '--threshold', default=0.95, type=float,
                        help='Similarity threshold for fuzzy matching (0-1, default: 0.85)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--log-file', type=str,
                        help='Log file path (default: classify_dois_TIMESTAMP.log)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING',
                                                'ERROR'], default='INFO', help='Logging level (default: INFO)')

    return parser.parse_args()


def tokenize_camelcase(text):
    result = re.sub('([a-z0-9])([A-Z])', r'\1 \2', text)
    result = re.sub('([A-Z]+)([A-Z][a-z])', r'\1 \2', result)
    return result.split()


def create_token_variants(tokens):
    if not tokens:
        return set()

    normalized_tokens = [t.lower() for t in tokens]

    variants = set()
    variants.add(''.join(normalized_tokens))

    if len(normalized_tokens) == 2:
        variants.add(normalized_tokens[0] + normalized_tokens[1])

    if len(normalized_tokens) == 3:
        variants.add(''.join(normalized_tokens))
        variants.add(normalized_tokens[0] +
                     normalized_tokens[1] + normalized_tokens[2])

    return variants


def smart_normalize(text, known_terms=None):
    if not text:
        return ""

    cleaned = text.lower()
    cleaned = re.sub(r'[-_\.\,\;\:\!\?\(\)\[\]\{\}\'\"\/\\]', ' ', cleaned)

    word_corrections = {
        'otput': 'output',
        'outpt': 'output',
        'ouput': 'output',
        'managment': 'management',
        'mangement': 'management',
        'managemnt': 'management',
        'artical': 'article',
        'articel': 'article',
        'jurnal': 'journal',
        'journel': 'journal',
        'confrence': 'conference',
        'conferance': 'conference',
        'papper': 'paper',
        'paer': 'paper',
        'disertation': 'dissertation',
        'dissertaion': 'dissertation',
        'colection': 'collection',
        'colletion': 'collection',
        'prepint': 'preprint',
        'pre-print': 'preprint',
        'standart': 'standard',
        'standar': 'standard',
        'servise': 'service',
        'servce': 'service',
        'datset': 'dataset',
        'data-set': 'dataset',
        'sofware': 'software',
        'sotfware': 'software',
        'modle': 'model',
        'modell': 'model',
        'reprot': 'report',
        'repotr': 'report',
        'reveiw': 'review',
        'reivew': 'review',
        'phisical': 'physical',
        'physcial': 'physical',
        'objet': 'object',
        'objct': 'object',
        'interative': 'interactive',
        'intractive': 'interactive',
        'resorce': 'resource',
        'resourse': 'resource',
        'registraton': 'registration',
        'registeration': 'registration',
        'audivisual': 'audiovisual',
        'audiovisal': 'audiovisual',
        'computaional': 'computational',
        'compuational': 'computational',
        'notbook': 'notebook',
        'notebbok': 'notebook',
        'instrumnt': 'instrument',
        'intrument': 'instrument',
        'awrad': 'award',
        'aword': 'award',
        'bok': 'book',
        'boook': 'book',
        'chaper': 'chapter',
        'chater': 'chapter',
        'imge': 'image',
        'inmage': 'image',
        'evnt': 'event',
        'evetn': 'event',
        'projet': 'project',
        'proejct': 'project',
        'sond': 'sound',
        'soudn': 'sound',
        'workfow': 'workflow',
        'work-flow': 'workflow',
        'txt': 'text',
        'texte': 'text',
        'otehr': 'other',
        'ohter': 'other',
        'stdy': 'study',
        'stuyd': 'study',
        'per': 'peer',
        'pear': 'peer',
    }

    words = cleaned.split()
    corrected_words = []
    for word in words:
        corrected = word_corrections.get(word, word)
        corrected_words.append(corrected)

    normalized = ''.join(corrected_words)

    return normalized


def normalize_text(text):
    return smart_normalize(text)


def load_reference_values(filepath):
    logger = logging.getLogger(__name__)
    normalized_values = set()
    normalized_to_original = {}

    try:
        logger.info(f"Loading reference values from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if isinstance(data, dict) and 'resource_type_general_values' in data:
            values = data['resource_type_general_values']
        elif isinstance(data, list):
            values = data
        else:
            logger.error(f"Invalid YAML structure in '{filepath}'")
            print(f"Error: Invalid YAML structure in '{filepath}'", file=sys.stderr)
            sys.exit(1)

        for original in tqdm(values, desc="Loading reference values", disable=len(values) < 50):
            if original:
                normalized = normalize_text(str(original))
                normalized_values.add(normalized)
                normalized_to_original[normalized] = original
                logger.debug(f"Loaded reference: {original} -> {normalized}")

        logger.info(f"Successfully loaded {len(normalized_values)} reference values")

    except FileNotFoundError:
        logger.error(f"Reference file '{filepath}' not found")
        print(f"Error: Reference file '{filepath}' not found", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading reference file: {e}")
        print(f"Error reading reference file: {e}", file=sys.stderr)
        sys.exit(1)

    return normalized_values, normalized_to_original


def fuzzy_match(text, reference_values, normalized_to_original, threshold=0.85):
    normalized_text = normalize_text(text)

    if normalized_text in reference_values:
        return normalized_to_original[normalized_text]

    tokens = text.split()
    if len(tokens) > 1:
        concatenated = ''.join([t.lower().strip() for t in tokens])
        concatenated = smart_normalize(concatenated)
        if concatenated in reference_values:
            return normalized_to_original[concatenated]

    camel_tokens = tokenize_camelcase(text)
    if len(camel_tokens) > 1:
        camel_normalized = smart_normalize(''.join(camel_tokens))
        if camel_normalized in reference_values:
            return normalized_to_original[camel_normalized]

    best_match = None
    best_ratio = 0

    for ref_value in reference_values:
        ratio = Levenshtein.ratio(normalized_text, ref_value)
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = ref_value

    if best_match:
        return normalized_to_original[best_match]

    return None


def process_csv(input_file, output_file, reference_file, chunksize=100000, threshold=0.85, verbose=False):
    logger = logging.getLogger(__name__)

    if verbose:
        print(f"Loading reference values from {reference_file}...")
    reference_values, normalized_to_original = load_reference_values(
        reference_file)
    if verbose:
        print(f"Loaded {len(reference_values)} reference values")

    doi_data = {}
    processed_rows = 0

    try:
        logger.info(f"Counting rows in {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for line in f) - 1

        logger.info(f"Processing {total_rows:,} rows from {input_file}")

        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            with tqdm(total=total_rows, desc="Processing CSV", unit="rows") as pbar:
                batch = []
                for row in reader:
                    batch.append(row)
                    processed_rows += 1

                    if len(batch) >= chunksize:
                        process_batch(batch, doi_data, reference_values,
                                      normalized_to_original, threshold)
                        pbar.update(len(batch))
                        logger.debug(f"Processed batch of {len(batch)} rows")
                        batch = []

                if batch:
                    process_batch(batch, doi_data, reference_values,
                                  normalized_to_original, threshold)
                    pbar.update(len(batch))
                    logger.debug(f"Processed final batch of {len(batch)} rows")

        logger.info(f"Total rows processed: {processed_rows:,}")
        logger.info(f"Unique DOIs found: {len(doi_data):,}")

        if verbose:
            print(f"Total rows processed: {processed_rows:,}")
            print(f"DOIs found: {len(doi_data):,}")

        write_output(output_file, doi_data, verbose)

    except FileNotFoundError:
        logger.error(f"Input file '{input_file}' not found")
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing CSV: {e}", exc_info=True)
        print(f"Error processing CSV: {e}", file=sys.stderr)
        sys.exit(1)


def process_batch(batch, doi_data, reference_values, normalized_to_original, threshold):
    for row in batch:
        doi = row.get('doi', '')
        subfield = row.get('subfield_path', '')
        value = row.get('value', '')

        if not doi:
            continue

        if doi not in doi_data:
            doi_data[doi] = {
                'resourceType': None,
                'resourceTypeGeneral': None,
                'inferredResourceTypeGeneral': None
            }

        if subfield == 'types.resourceType':
            doi_data[doi]['resourceType'] = value
        elif subfield == 'types.resourceTypeGeneral':
            doi_data[doi]['resourceTypeGeneral'] = value


def write_output(output_file, doi_data, verbose):
    logger = logging.getLogger(__name__)

    reference_values, normalized_to_original = load_reference_values(
        parse_arguments().reference
    )
    threshold = parse_arguments().threshold

    output_count = 0
    matched_count = 0
    text_other_count = 0
    excluded_count = 0

    logger.info(f"Writing output to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        fieldnames = ['doi', 'resourceType', 'resourceTypeGeneral',
                      'inferredResourceTypeGeneral']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        filtered_dois = {
            doi: data for doi, data in doi_data.items()
            if data['resourceTypeGeneral'] in ['Text', 'Other']
        }
        text_other_count = len(filtered_dois)

        logger.info(f"Found {text_other_count:,} DOIs with resourceTypeGeneral='Text' or 'Other'")

        for doi, data in tqdm(filtered_dois.items(), desc="Matching DOIs", unit="DOI"):
            if data['resourceType']:
                match = fuzzy_match(data['resourceType'], reference_values,
                                    normalized_to_original, threshold)
                if match:
                    normalized_resource_type = normalize_text(
                        data['resourceType']).lower()

                    if ((normalized_resource_type in ['text', 'txt'] and match in ['Text', 'Other']) or
                            (normalized_resource_type in ['other', 'otehr', 'ohter'] and match in ['Text', 'Other'])):
                        excluded_count += 1
                        logger.debug(f"Excluded redundant classification: {data['resourceType']} -> {match}")
                        continue

                    data['inferredResourceTypeGeneral'] = match
                    writer.writerow({
                        'doi': doi,
                        'resourceType': data['resourceType'],
                        'resourceTypeGeneral': data['resourceTypeGeneral'],
                        'inferredResourceTypeGeneral': match
                    })
                    output_count += 1
                    matched_count += 1
                    logger.debug(f"Matched: {data['resourceType']} -> {match}")
                else:
                    logger.debug(f"No match for: {data['resourceType']}")

    logger.info(f"Output statistics:")
    logger.info(f"  - Total DOIs processed: {len(doi_data):,}")
    logger.info(f"  - DOIs with Text/Other: {text_other_count:,}")
    logger.info(f"  - DOIs matched: {matched_count:,}")
    logger.info(f"  - DOIs excluded (redundant): {excluded_count:,}")
    logger.info(f"  - DOIs written to output: {output_count:,}")
    logger.info(f"  - Match rate: {matched_count/text_other_count*100:.1f}%" if text_other_count > 0 else "N/A")

    if verbose:
        print(f"Output written to {output_file}")
        print(f"Total DOIs written: {output_count:,}")
        print(f"DOIs excluded (redundant): {excluded_count:,}")


def setup_logging(log_file=None, log_level='INFO'):
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'classify_dois_{timestamp}.log'

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    args = parse_arguments()

    logger = setup_logging(args.log_file, args.log_level)
    logger.info(f"Starting DOI classification script")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Reference file: {args.reference}")
    logger.info(f"Chunk size: {args.chunksize:,}")
    logger.info(f"Similarity threshold: {args.threshold}")

    if args.verbose:
        print(f"Processing {args.input}")
        print(f"Reference file: {args.reference}")
        print(f"Output file: {args.output}")
        print(f"Chunk size: {args.chunksize:,}")
        print(f"Similarity threshold: {args.threshold}")
        if args.log_file:
            print(f"Log file: {args.log_file}")

    try:
        process_csv(
            input_file=args.input,
            output_file=args.output,
            reference_file=args.reference,
            chunksize=args.chunksize,
            threshold=args.threshold,
            verbose=args.verbose
        )

        logger.info("Processing completed successfully")
        if args.verbose:
            print("Processing complete!")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
