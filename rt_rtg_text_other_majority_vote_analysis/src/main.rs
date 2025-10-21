use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::bounded;
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use log::info;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::thread;
use strsim::normalized_levenshtein;

static WORD_CORRECTIONS: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("otput", "output");
    map.insert("outpt", "output");
    map.insert("ouput", "output");
    map.insert("managment", "management");
    map.insert("mangement", "management");
    map.insert("managemnt", "management");
    map.insert("artical", "article");
    map.insert("articel", "article");
    map.insert("jurnal", "journal");
    map.insert("journel", "journal");
    map.insert("confrence", "conference");
    map.insert("conferance", "conference");
    map.insert("papper", "paper");
    map.insert("paer", "paper");
    map.insert("disertation", "dissertation");
    map.insert("dissertaion", "dissertation");
    map.insert("colection", "collection");
    map.insert("colletion", "collection");
    map.insert("prepint", "preprint");
    map.insert("standart", "standard");
    map.insert("standar", "standard");
    map.insert("servise", "service");
    map.insert("servce", "service");
    map.insert("datset", "dataset");
    map.insert("sofware", "software");
    map.insert("sotfware", "software");
    map.insert("modle", "model");
    map.insert("modell", "model");
    map.insert("reprot", "report");
    map.insert("repotr", "report");
    map.insert("reveiw", "review");
    map.insert("reivew", "review");
    map.insert("phisical", "physical");
    map.insert("physcial", "physical");
    map.insert("objet", "object");
    map.insert("objct", "object");
    map.insert("interative", "interactive");
    map.insert("intractive", "interactive");
    map.insert("resorce", "resource");
    map.insert("resourse", "resource");
    map.insert("registraton", "registration");
    map.insert("registeration", "registration");
    map.insert("audivisual", "audiovisual");
    map.insert("audiovisal", "audiovisual");
    map.insert("computaional", "computational");
    map.insert("compuational", "computational");
    map.insert("notbook", "notebook");
    map.insert("notebbok", "notebook");
    map.insert("instrumnt", "instrument");
    map.insert("intrument", "instrument");
    map.insert("awrad", "award");
    map.insert("aword", "award");
    map.insert("bok", "book");
    map.insert("boook", "book");
    map.insert("chaper", "chapter");
    map.insert("chater", "chapter");
    map.insert("imge", "image");
    map.insert("inmage", "image");
    map.insert("evnt", "event");
    map.insert("evetn", "event");
    map.insert("projet", "project");
    map.insert("proejct", "project");
    map.insert("sond", "sound");
    map.insert("soudn", "sound");
    map.insert("workfow", "workflow");
    map.insert("txt", "text");
    map.insert("texte", "text");
    map.insert("otehr", "other");
    map.insert("ohter", "other");
    map.insert("stdy", "study");
    map.insert("stuyd", "study");
    map.insert("per", "peer");
    map.insert("pear", "peer");
    map
});

const CHANNEL_BUFFER: usize = 16_384;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(long, default_value_t = 5)]
    min_count: u64,
    #[arg(long, default_value_t = 0.86)]
    match_threshold: f64,
    #[arg(long, default_value_t = 0.55)]
    majority_threshold: f64,
    #[arg(long, default_value_t = 3)]
    max_candidates: usize,
    #[arg(long)]
    allow_other_to_text: bool,
    #[arg(long, default_value_t = String::from("info"))]
    log_level: String,
}

#[derive(Debug)]
struct CompletedRecord {
    resource_type: String,
    resource_type_general: String,
}

#[derive(Default)]
struct ResourceAggregate {
    general_counts: HashMap<String, u64>,
    original_forms: HashMap<String, u64>,
    examples_by_general: HashMap<String, HashMap<String, u64>>,
}

impl ResourceAggregate {
    fn add(&mut self, resource_type: &str, general: &str) {
        *self
            .original_forms
            .entry(resource_type.to_string())
            .or_insert(0) += 1;

        let general_key = general.trim();
        if general_key.is_empty() {
            return;
        }

        let general_key = general_key.to_string();

        *self.general_counts.entry(general_key.clone()).or_insert(0) += 1;

        let examples = self.examples_by_general.entry(general_key).or_default();
        *examples.entry(resource_type.to_string()).or_insert(0) += 1;
    }
}

#[derive(Default)]
struct Aggregator {
    aggregates: HashMap<String, ResourceAggregate>,
}

impl Aggregator {
    fn add_record(&mut self, record: CompletedRecord) {
        let normalized = normalize_for_key(&record.resource_type);
        if normalized.is_empty() {
            return;
        }
        let entry = self.aggregates.entry(normalized).or_default();
        entry.add(&record.resource_type, &record.resource_type_general);
    }

    fn unique_keys(&self) -> usize {
        self.aggregates.len()
    }

    fn source_key_count(&self, source_general: &str, min_count: u64) -> usize {
        self.aggregates
            .values()
            .filter(|agg| count_for_general(&agg.general_counts, source_general) >= min_count)
            .count()
    }
}

#[derive(Debug)]
struct Suggestion {
    normalized_key: String,
    representative: String,
    source_general: String,
    source_count: u64,
    majority_general: String,
    majority_ratio: f64,
    candidate_total: u64,
    matched_variant_count: usize,
    matched_examples: Vec<String>,
    source_examples: Vec<String>,
    ranked_candidates: Vec<(String, u64, f64)>,
}

#[derive(Default)]
struct PendingRecord {
    resource_type: Option<String>,
    general: Option<String>,
}

#[derive(Default)]
struct ProducerStats {
    rows_seen: u64,
    completed_dois: u64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:?}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    init_logging(&args.log_level)?;

    let (sender, receiver) = bounded(CHANNEL_BUFFER);
    let input_path = args.input.clone();

    let producer = thread::spawn(move || produce_records(&input_path, sender));

    let mut aggregator = Aggregator::default();
    for record in receiver.iter() {
        aggregator.add_record(record);
    }

    let producer_stats = producer
        .join()
        .expect("producer thread panicked")
        .with_context(|| "failed to read input")?;

    info!(
        "Processed {} rows and {} DOIs with both resourceType values",
        producer_stats.rows_seen, producer_stats.completed_dois
    );
    info!(
        "Tracked {} unique normalized resourceType values; Other meet threshold: {}, Text meet threshold: {}",
        aggregator.unique_keys(),
        aggregator.source_key_count("Other", args.min_count),
        aggregator.source_key_count("Text", args.min_count)
    );

    let suggestions = compute_suggestions(
        &aggregator,
        args.min_count,
        args.match_threshold,
        args.majority_threshold,
        args.max_candidates,
        args.allow_other_to_text,
    );

    info!("Prepared {} candidate reclassifications", suggestions.len());

    write_output(&args.output, &suggestions, args.max_candidates)?;

    Ok(())
}

fn init_logging(level: &str) -> Result<()> {
    let env_level = match level.to_ascii_lowercase().as_str() {
        "debug" => "debug",
        "warn" => "warn",
        "error" => "error",
        _ => "info",
    };
    std::env::set_var("RUST_LOG", env_level);
    env_logger::builder()
        .format_timestamp_secs()
        .try_init()
        .or_else(|_| {
            env_logger::builder()
                .format_timestamp_secs()
                .is_test(true)
                .try_init()
        })
        .context("failed to initialise logger")
}

fn produce_records(
    path: &Path,
    sender: crossbeam_channel::Sender<CompletedRecord>,
) -> Result<ProducerStats> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("failed to open {:?}", path))?;

    let headers = reader.headers().context("missing CSV headers")?.clone();

    let doi_idx = find_index(&headers, "doi")?;
    let subfield_idx = find_index(&headers, "subfield_path")?;
    let value_idx = find_index(&headers, "value")?;

    let mut pending: HashMap<String, PendingRecord> = HashMap::new();
    let mut stats = ProducerStats::default();

    for result in reader.records() {
        stats.rows_seen += 1;
        let record = result?;
        handle_record(
            &record,
            doi_idx,
            subfield_idx,
            value_idx,
            &mut pending,
            &sender,
            &mut stats,
        )?;
    }

    for record in pending.into_values() {
        if let (Some(resource_type), Some(general)) = (record.resource_type, record.general) {
            let completed = CompletedRecord {
                resource_type,
                resource_type_general: general,
            };
            sender.send(completed).ok();
            stats.completed_dois += 1;
        }
    }

    Ok(stats)
}

fn handle_record(
    record: &StringRecord,
    doi_idx: usize,
    subfield_idx: usize,
    value_idx: usize,
    pending: &mut HashMap<String, PendingRecord>,
    sender: &crossbeam_channel::Sender<CompletedRecord>,
    stats: &mut ProducerStats,
) -> Result<()> {
    let doi = record.get(doi_idx).unwrap_or_default().trim().to_string();
    if doi.is_empty() {
        return Ok(());
    }

    let subfield = record.get(subfield_idx).unwrap_or_default().trim();
    if subfield != "types.resourceType" && subfield != "types.resourceTypeGeneral" {
        return Ok(());
    }

    let value = record.get(value_idx).unwrap_or_default().trim().to_string();

    let ready = {
        let entry = pending.entry(doi.clone()).or_default();
        match subfield {
            "types.resourceType" if entry.resource_type.is_none() => {
                entry.resource_type = Some(value);
            }
            "types.resourceTypeGeneral" if entry.general.is_none() => {
                entry.general = Some(value);
            }
            _ => {}
        }
        entry.resource_type.is_some() && entry.general.is_some()
    };

    if ready {
        if let Some(ready_record) = pending.remove(&doi) {
            if let (Some(resource_type), Some(general)) =
                (ready_record.resource_type, ready_record.general)
            {
                let completed = CompletedRecord {
                    resource_type,
                    resource_type_general: general,
                };
                sender.send(completed)?;
                stats.completed_dois += 1;
            }
        }
    }

    Ok(())
}

fn find_index(headers: &StringRecord, name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|header| header == name)
        .with_context(|| format!("missing required column '{name}'"))
}

fn is_delimiter(ch: char) -> bool {
    matches!(
        ch,
        '-' | '_'
            | '.'
            | ','
            | ';'
            | ':'
            | '!'
            | '?'
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '\''
            | '"'
            | '/'
            | '\\'
    )
}

fn smart_normalize(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let lower: String = text
        .chars()
        .map(|ch| {
            let ch_lower = ch.to_ascii_lowercase();
            if is_delimiter(ch_lower) {
                ' '
            } else {
                ch_lower
            }
        })
        .collect();

    let mut normalized = String::new();
    for token in lower.split_whitespace() {
        if let Some(replacement) = WORD_CORRECTIONS.get(token) {
            normalized.push_str(replacement);
        } else {
            normalized.push_str(token);
        }
    }

    normalized
}

fn normalize_for_key(text: &str) -> String {
    smart_normalize(text.trim())
}

fn compute_suggestions(
    aggregator: &Aggregator,
    min_count: u64,
    match_threshold: f64,
    majority_threshold: f64,
    max_candidates: usize,
    allow_other_to_text: bool,
) -> Vec<Suggestion> {
    if aggregator.aggregates.is_empty() {
        return Vec::new();
    }

    let keys: Vec<&String> = aggregator.aggregates.keys().collect();

    struct SourceConfig<'a> {
        general: &'a str,
        excludes: Vec<&'a str>,
    }

    let mut configs = Vec::new();
    let mut other_excludes = vec!["Other"];
    if !allow_other_to_text {
        other_excludes.push("Text");
    }
    configs.push(SourceConfig {
        general: "Other",
        excludes: other_excludes,
    });
    configs.push(SourceConfig {
        general: "Text",
        excludes: vec!["Text", "Other"],
    });

    let mut suggestions = Vec::new();

    for (normalized_key, aggregate) in &aggregator.aggregates {
        for config in &configs {
            let source_count = count_for_general(&aggregate.general_counts, config.general);
            if source_count < min_count {
                continue;
            }

            if let Some(suggestion) = evaluate_source_general(
                normalized_key,
                aggregate,
                &aggregator.aggregates,
                &keys,
                config.general,
                source_count,
                match_threshold,
                majority_threshold,
                max_candidates,
                &config.excludes,
            ) {
                suggestions.push(suggestion);
            }
        }
    }

    suggestions.sort_by(|a, b| {
        b.source_count
            .cmp(&a.source_count)
            .then_with(|| {
                b.majority_ratio
                    .partial_cmp(&a.majority_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.majority_general.cmp(&b.majority_general))
    });

    suggestions
}

#[allow(clippy::too_many_arguments)]
fn evaluate_source_general(
    normalized_key: &str,
    aggregate: &ResourceAggregate,
    aggregates: &HashMap<String, ResourceAggregate>,
    keys: &[&String],
    source_general: &str,
    source_count: u64,
    match_threshold: f64,
    majority_threshold: f64,
    max_candidates: usize,
    exclude_generals: &[&str],
) -> Option<Suggestion> {
    if source_count == 0 {
        return None;
    }

    let matches: Vec<(&String, f64)> = keys
        .par_iter()
        .map(|candidate| {
            let score = normalized_levenshtein(normalized_key, candidate.as_str());
            (*candidate, score)
        })
        .filter(|(_, score)| *score >= match_threshold)
        .collect();

    if matches.is_empty() {
        return None;
    }

    let mut totals: HashMap<String, (String, u64)> = HashMap::new();
    for (candidate_key, _) in &matches {
        if let Some(candidate_agg) = aggregates.get(*candidate_key) {
            for (general, count) in &candidate_agg.general_counts {
                let general_trimmed = general.trim();
                if general_trimmed.is_empty()
                    || general_trimmed.eq_ignore_ascii_case(source_general)
                    || exclude_generals
                        .iter()
                        .any(|excluded| general_trimmed.eq_ignore_ascii_case(excluded))
                {
                    continue;
                }
                let canonical = canonical_general(general_trimmed);
                let entry = totals
                    .entry(canonical)
                    .or_insert_with(|| (general.clone(), 0));
                entry.1 += *count;
            }
        }
    }

    if totals.is_empty() {
        return None;
    }

    let candidate_total: u64 = totals.values().map(|(_, count)| *count).sum();
    if candidate_total == 0 {
        return None;
    }

    let mut ranked: Vec<(String, u64)> = totals
        .values()
        .map(|(display, count)| (display.clone(), *count))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let (majority_general, majority_votes) = ranked[0].clone();
    let majority_ratio = majority_votes as f64 / candidate_total as f64;
    if majority_ratio < majority_threshold {
        return None;
    }

    let matched_variant_count = matches.len();

    let matched_examples: Vec<String> = matches
        .iter()
        .take(max_candidates)
        .map(|(key, score)| {
            let example = aggregates
                .get(*key)
                .and_then(|agg| agg.original_forms.iter().max_by(|a, b| a.1.cmp(b.1)))
                .map(|(value, _)| value.clone())
                .unwrap_or_else(|| (*key).clone());
            format!("{example} ({score:.2})")
        })
        .collect();

    let combined_examples =
        aggregate_examples_for_general(&aggregate.examples_by_general, source_general);
    let source_examples: Vec<String> = top_n(&combined_examples, 3)
        .into_iter()
        .map(|(value, _)| value)
        .collect();

    let ranked_candidates: Vec<(String, u64, f64)> = ranked
        .into_iter()
        .take(max_candidates)
        .map(|(general, count)| {
            let ratio = count as f64 / candidate_total as f64;
            (general, count, ratio)
        })
        .collect();

    let representative = source_examples.first().cloned().unwrap_or_else(String::new);

    Some(Suggestion {
        normalized_key: normalized_key.to_string(),
        representative,
        source_general: source_general.to_string(),
        source_count,
        majority_general,
        majority_ratio,
        candidate_total,
        matched_variant_count,
        matched_examples,
        source_examples,
        ranked_candidates,
    })
}

fn canonical_general(general: &str) -> String {
    general.trim().to_ascii_lowercase()
}

fn count_for_general(counts: &HashMap<String, u64>, general: &str) -> u64 {
    counts
        .iter()
        .filter(|(key, _)| key.trim().eq_ignore_ascii_case(general))
        .map(|(_, count)| *count)
        .sum()
}

fn aggregate_examples_for_general(
    examples: &HashMap<String, HashMap<String, u64>>,
    general: &str,
) -> HashMap<String, u64> {
    let mut combined = HashMap::new();
    for (key, forms) in examples {
        if key.trim().eq_ignore_ascii_case(general) {
            for (value, count) in forms {
                *combined.entry(value.clone()).or_insert(0) += *count;
            }
        }
    }
    combined
}

fn top_n(counter: &HashMap<String, u64>, n: usize) -> Vec<(String, u64)> {
    let mut pairs: Vec<(String, u64)> = counter
        .iter()
        .map(|(value, count)| (value.clone(), *count))
        .collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs.truncate(n);
    pairs
}

fn write_output(path: &Path, suggestions: &[Suggestion], max_candidates: usize) -> Result<()> {
    let file = File::create(path).with_context(|| format!("failed to create {:?}", path))?;
    let mut writer = WriterBuilder::new().from_writer(file);

    let mut headers = vec![
        "normalized_resource_type".to_string(),
        "representative_resource_type".to_string(),
        "source_resourceTypeGeneral".to_string(),
        "source_doi_count".to_string(),
        "majority_resourceTypeGeneral".to_string(),
        "majority_ratio".to_string(),
        "candidate_total".to_string(),
        "matched_variant_count".to_string(),
        "matched_examples".to_string(),
        "source_examples".to_string(),
    ];

    for idx in 1..=max_candidates {
        headers.push(format!("candidate_{idx}_resourceTypeGeneral"));
        headers.push(format!("candidate_{idx}_count"));
        headers.push(format!("candidate_{idx}_ratio"));
    }
    writer.write_record(&headers)?;

    for suggestion in suggestions {
        let mut row = vec![
            suggestion.normalized_key.clone(),
            suggestion.representative.clone(),
            suggestion.source_general.clone(),
            suggestion.source_count.to_string(),
            suggestion.majority_general.clone(),
            format!("{:.3}", suggestion.majority_ratio),
            suggestion.candidate_total.to_string(),
            suggestion.matched_variant_count.to_string(),
            suggestion.matched_examples.join("; "),
            suggestion.source_examples.join("; "),
        ];

        for idx in 0..max_candidates {
            if let Some((general, count, ratio)) = suggestion.ranked_candidates.get(idx) {
                row.push(general.clone());
                row.push(count.to_string());
                row.push(format!("{ratio:.3}"));
            } else {
                row.push(String::new());
                row.push(String::new());
                row.push(String::new());
            }
        }

        writer.write_record(row)?;
    }

    writer.flush()?;
    Ok(())
}
