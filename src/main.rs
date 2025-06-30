use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufReader, Cursor, Read},
    path::Path,
    sync::Arc,
};
use arrow::{
    array::{Float64Array, TimestampMillisecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use byteorder::{BigEndian, ReadBytesExt};
use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use lzma_rs::lzma_decompress;
use parquet::{
    arrow::ArrowWriter,
    basic::Compression,
    file::properties::WriterProperties,
};
use rayon::prelude::*;

#[derive(Debug, Clone)]
struct Tick {
    timestamp_ms: u64,
    bid: f64,
    ask: f64,
    volume: f64,
}

#[derive(Debug, Clone)]
struct OHLCVBar {
    timestamp: DateTime<Utc>,
    open_bid: f64,
    high_bid: f64,
    low_bid: f64,
    close_bid: f64,
    open_ask: f64,
    high_ask: f64,
    low_ask: f64,
    close_ask: f64,
    volume: f64,
    tick_count: u64,
    spread_avg: f64,
    spread_max: f64,
    spread_min: f64,
    vwap_bid: f64,
    vwap_ask: f64,
}
#[warn(dead_code)]
#[derive(Debug, Clone)]
enum TimeFrame {
    Seconds(u32),
    Minutes(u32),
}

impl TimeFrame {
    fn to_seconds(&self) -> u32 {
        match self {
            TimeFrame::Seconds(s) => *s,
            TimeFrame::Minutes(m) => m * 60,
        }
    }
}

fn normalize_price(price: u32, symbol: &str) -> f64 {
    match symbol {
        "EURUSD" | "GBPUSD" | "USDCHF" => price as f64 / 100_000.0,
        "USDJPY" => price as f64 / 1_000.0,
        _ => {
            eprintln!("⚠️ Unknown symbol {}, using default normalization (1e5)", symbol);
            price as f64 / 100_000.0
        }
    }
}

fn parse_bi5(data: &[u8], symbol: &str) -> Vec<Tick> {
    let mut cursor = Cursor::new(data);
    let mut ticks = Vec::with_capacity(1000);
    let mut invalid_count = 0;

    while let Ok(ts) = cursor.read_u32::<BigEndian>() {
        let bid_points = cursor.read_u32::<BigEndian>().unwrap_or(0);
        let ask_points = cursor.read_u32::<BigEndian>().unwrap_or(0);
        let bid_volume = cursor.read_f32::<BigEndian>().unwrap_or(0.0);
        let ask_volume = cursor.read_f32::<BigEndian>().unwrap_or(0.0);

        let mut bid = normalize_price(bid_points, symbol);
        let mut ask = normalize_price(ask_points, symbol);
        let volume = (bid_volume + ask_volume) as f64;

        if bid > 0.0 && ask > 0.0 && ask <= bid {
            std::mem::swap(&mut bid, &mut ask);
            if invalid_count < 10 {
                eprintln!("⚠️ Swapped bid/ask for ts={} (bid={:.5}, ask={:.5}, volume={})", ts, bid, ask, volume);
            }
            invalid_count += 1;
        }

        if bid > 0.0 && ask > 0.0 && ask > bid && volume > 0.0 {
            ticks.push(Tick {
                timestamp_ms: ts as u64,
                bid,
                ask,
                volume,
            });
        } else {
            if invalid_count < 10 {
                eprintln!("⚠️ Invalid tick: ts={}, bid={:.5}, ask={:.5}, volume={}", ts, bid, ask, volume);
            }
            invalid_count += 1;
        }
    }

    if ticks.is_empty() {
        eprintln!("⚠️ No valid ticks parsed from data (length: {} bytes, {} invalid ticks)", data.len(), invalid_count);
    } else if invalid_count > 0 {
        eprintln!("⚠️ Skipped {} invalid ticks from data (length: {} bytes)", invalid_count, data.len());
    }
    ticks
}

fn parse_text_ticks(data: &str, _symbol: &str) -> Vec<Tick> {
    let mut ticks = Vec::with_capacity(1000);
    let mut seen = HashSet::new();
    let mut invalid_count = 0;

    for line in data.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            if invalid_count < 10 {
                eprintln!("⚠️ Skipping malformed line: {}", line);
            }
            invalid_count += 1;
            continue;
        }

        let timestamp_str = parts[0].trim_start_matches("Time: ");
        let timestamp = match DateTime::parse_from_str(timestamp_str, "%Y-%m-%d %H:%M:%S%.f%z") {
            Ok(dt) => dt,
            Err(e) => {
                if invalid_count < 10 {
                    eprintln!("⚠️ Invalid timestamp in line '{}': {}", line, e);
                }
                invalid_count += 1;
                continue;
            }
        };
        let bid: f64 = match parts[1].trim_start_matches("Bid: ").parse() {
            Ok(b) => b,
            Err(e) => {
                if invalid_count < 10 {
                    eprintln!("⚠️ Invalid bid in line '{}': {}", line, e);
                }
                invalid_count += 1;
                continue;
            }
        };
        let ask: f64 = match parts[2].trim_start_matches("Ask: ").parse() {
            Ok(a) => a,
            Err(e) => {
                if invalid_count < 10 {
                    eprintln!("⚠️ Invalid ask in line '{}': {}", line, e);
                }
                invalid_count += 1;
                continue;
            }
        };
        let volume: f64 = match parts[3].trim_start_matches("Volume: ").parse() {
            Ok(v) => v,
            Err(e) => {
                if invalid_count < 10 {
                    eprintln!("⚠️ Invalid volume in line '{}': {}", line, e);
                }
                invalid_count += 1;
                continue;
            }
        };

        if bid > 0.0 && ask > 0.0 && ask > bid && volume > 0.0 {
            let tick = Tick {
                timestamp_ms: timestamp.timestamp_millis() as u64,
                bid,
                ask,
                volume,
            };
            let key = (
                tick.timestamp_ms,
                (tick.bid * 100_000.0) as u64,
                (tick.ask * 100_000.0) as u64,
                (tick.volume * 100.0) as u64,
            );
            if seen.insert(key) {
                ticks.push(tick);
            }
        } else {
            if invalid_count < 10 {
                eprintln!("⚠️ Invalid tick: bid={:.5}, ask={:.5}, volume={}", bid, ask, volume);
            }
            invalid_count += 1;
        }
    }

    if invalid_count > 0 {
        eprintln!("⚠️ Skipped {} invalid ticks from text data", invalid_count);
    }
    ticks
}

fn format_timestamp(base_date: &str, ms: u64) -> DateTime<Utc> {
    let base = NaiveDate::parse_from_str(base_date, "%Y%m%d")
        .unwrap_or_else(|_| NaiveDate::from_ymd_opt(2000, 1, 1).unwrap())
        .and_hms_opt(0, 0, 0)
        .unwrap();
    Utc.from_utc_datetime(&(base + Duration::milliseconds(ms as i64)))
}

fn extract_date_from_filename(file_name: &str) -> Option<String> {
    let parts: Vec<&str> = file_name.split('_').collect();
    if parts.len() >= 3 {
        Some(parts[1].to_string())
    } else {
        None
    }
}

fn aggregate_ticks_to_ohlcv(ticks: &[Tick], base_date: &str, timeframe: &TimeFrame, writer: &mut ArrowWriter<File>, schema: &Arc<Schema>) {
    if ticks.is_empty() {
        return;
    }

    let interval_seconds = timeframe.to_seconds();
    let mut bars: HashMap<i64, Vec<Tick>> = HashMap::new();

    for tick in ticks {
        let timestamp = format_timestamp(base_date, tick.timestamp_ms);
        let interval_key = timestamp.timestamp() / interval_seconds as i64;
        bars.entry(interval_key).or_insert_with(|| Vec::with_capacity(100)).push(tick.clone());
    }

    let mut result = Vec::with_capacity(bars.len());
    for (interval_key, interval_ticks) in bars {
        if interval_ticks.is_empty() {
            continue;
        }

        let bar_timestamp = Utc.timestamp_opt(interval_key * interval_seconds as i64, 0).unwrap();

        let open_bid = interval_ticks[0].bid;
        let close_bid = interval_ticks[interval_ticks.len() - 1].bid;
        let high_bid = interval_ticks.iter().map(|t| t.bid).fold(f64::NEG_INFINITY, f64::max);
        let low_bid = interval_ticks.iter().map(|t| t.bid).fold(f64::INFINITY, f64::min);

        let open_ask = interval_ticks[0].ask;
        let close_ask = interval_ticks[interval_ticks.len() - 1].ask;
        let high_ask = interval_ticks.iter().map(|t| t.ask).fold(f64::NEG_INFINITY, f64::max);
        let low_ask = interval_ticks.iter().map(|t| t.ask).fold(f64::INFINITY, f64::min);

        let total_volume: f64 = interval_ticks.iter().map(|t| t.volume).sum();
        let tick_count = interval_ticks.len() as u64;

        let spreads: Vec<f64> = interval_ticks.iter().map(|t| t.ask - t.bid).collect();
        let spread_avg = if !spreads.is_empty() {
            spreads.iter().sum::<f64>() / spreads.len() as f64
        } else {
            0.0
        };
        let spread_max = spreads.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let spread_min = spreads.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let total_volume_weighted_bid: f64 = interval_ticks.iter().map(|t| t.bid * t.volume).sum();
        let total_volume_weighted_ask: f64 = interval_ticks.iter().map(|t| t.ask * t.volume).sum();
        let vwap_bid = if total_volume > 0.0 { total_volume_weighted_bid / total_volume } else { close_bid };
        let vwap_ask = if total_volume > 0.0 { total_volume_weighted_ask / total_volume } else { close_ask };

        result.push(OHLCVBar {
            timestamp: bar_timestamp,
            open_bid,
            high_bid,
            low_bid,
            close_bid,
            open_ask,
            high_ask,
            low_ask,
            close_ask,
            volume: total_volume,
            tick_count,
            spread_avg,
            spread_max,
            spread_min,
            vwap_bid,
            vwap_ask,
        });
    }

    result.sort_by_key(|bar| bar.timestamp);

    if result.is_empty() {
        return;
    }

    let chunk_size = 10_000;
    for chunk in result.chunks(chunk_size) {
        let mut chunk_bars = chunk.to_vec();
        preprocess_bars(&mut chunk_bars);

        let timestamps: Vec<i64> = chunk_bars.iter().map(|bar| bar.timestamp.timestamp_millis()).collect();
        let open_bids: Vec<f64> = chunk_bars.iter().map(|bar| bar.open_bid).collect();
        let high_bids: Vec<f64> = chunk_bars.iter().map(|bar| bar.high_bid).collect();
        let low_bids: Vec<f64> = chunk_bars.iter().map(|bar| bar.low_bid).collect();
        let close_bids: Vec<f64> = chunk_bars.iter().map(|bar| bar.close_bid).collect();
        let open_asks: Vec<f64> = chunk_bars.iter().map(|bar| bar.open_ask).collect();
        let high_asks: Vec<f64> = chunk_bars.iter().map(|bar| bar.high_ask).collect();
        let low_asks: Vec<f64> = chunk_bars.iter().map(|bar| bar.low_ask).collect();
        let close_asks: Vec<f64> = chunk_bars.iter().map(|bar| bar.close_ask).collect();
        let volumes: Vec<f64> = chunk_bars.iter().map(|bar| bar.volume).collect();
        let tick_counts: Vec<u64> = chunk_bars.iter().map(|bar| bar.tick_count).collect();
        let spread_avgs: Vec<f64> = chunk_bars.iter().map(|bar| bar.spread_avg).collect();
        let spread_maxs: Vec<f64> = chunk_bars.iter().map(|bar| bar.spread_max).collect();
        let spread_mins: Vec<f64> = chunk_bars.iter().map(|bar| bar.spread_min).collect();
        let vwap_bids: Vec<f64> = chunk_bars.iter().map(|bar| bar.vwap_bid).collect();
        let vwap_asks: Vec<f64> = chunk_bars.iter().map(|bar| bar.vwap_ask).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(TimestampMillisecondArray::from(timestamps)),
                Arc::new(Float64Array::from(open_bids)),
                Arc::new(Float64Array::from(high_bids)),
                Arc::new(Float64Array::from(low_bids)),
                Arc::new(Float64Array::from(close_bids)),
                Arc::new(Float64Array::from(open_asks)),
                Arc::new(Float64Array::from(high_asks)),
                Arc::new(Float64Array::from(low_asks)),
                Arc::new(Float64Array::from(close_asks)),
                Arc::new(Float64Array::from(volumes)),
                Arc::new(UInt64Array::from(tick_counts)),
                Arc::new(Float64Array::from(spread_avgs)),
                Arc::new(Float64Array::from(spread_maxs)),
                Arc::new(Float64Array::from(spread_mins)),
                Arc::new(Float64Array::from(vwap_bids)),
                Arc::new(Float64Array::from(vwap_asks)),
            ],
        )
        .expect("Failed to create record batch");

        writer.write(&batch).expect("Failed to write batch");
    }
}

fn create_ibkr_compatible_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("datetime", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
        Field::new("open_bid", DataType::Float64, false),
        Field::new("high_bid", DataType::Float64, false),
        Field::new("low_bid", DataType::Float64, false),
        Field::new("close_bid", DataType::Float64, false),
        Field::new("open_ask", DataType::Float64, false),
        Field::new("high_ask", DataType::Float64, false),
        Field::new("low_ask", DataType::Float64, false),
        Field::new("close_ask", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
        Field::new("tick_count", DataType::UInt64, false),
        Field::new("spread_avg", DataType::Float64, false),
        Field::new("spread_max", DataType::Float64, false),
        Field::new("spread_min", DataType::Float64, false),
        Field::new("vwap_bid", DataType::Float64, false),
        Field::new("vwap_ask", DataType::Float64, false),
    ]))
}

fn preprocess_bars(bars: &mut Vec<OHLCVBar>) {
    for bar in bars.iter_mut() {
        if bar.spread_avg.is_nan() || bar.spread_avg.is_infinite() {
            bar.spread_avg = 0.0;
        }
        if bar.spread_max.is_nan() || bar.spread_max.is_infinite() {
            bar.spread_max = bar.spread_avg;
        }
        if bar.spread_min.is_nan() || bar.spread_min.is_infinite() {
            bar.spread_min = bar.spread_avg;
        }
        if bar.vwap_bid.is_nan() || bar.vwap_bid.is_infinite() {
            bar.vwap_bid = bar.close_bid;
        }
        if bar.vwap_ask.is_nan() || bar.vwap_ask.is_infinite() {
            bar.vwap_ask = bar.close_ask;
        }
    }
}

fn convert_to_training_data(
    file_paths: &[&Path],
    symbol: &str,
    output_dir: &str,
    timeframe: TimeFrame,
) {
    let schema = create_ibkr_compatible_schema();
    let timeframe_str = match timeframe {
        TimeFrame::Seconds(s) => format!("{}s", s),
        TimeFrame::Minutes(m) => format!("{}m", m),
    };
    let output_path = Path::new(output_dir).join(format!("{}_{}.parquet", symbol, timeframe_str));

    let mut writer = match File::create(&output_path) {
        Ok(file) => Some(ArrowWriter::try_new(
            file,
            schema.clone(),
            Some(WriterProperties::builder().set_compression(Compression::SNAPPY).build()),
        ).expect("Failed to create Parquet writer")),
        Err(e) => {
            eprintln!("❌ Failed to create Parquet file {}: {}", output_path.display(), e);
            return;
        }
    };

    let mut total_bars = 0;

    for &file_path in file_paths {
        let file_name = file_path.file_name().unwrap().to_string_lossy();
        let base_date = extract_date_from_filename(&file_name).unwrap_or_else(|| "20000101".into());

        let ticks = if file_path.extension().map(|e| e == "txt").unwrap_or(false) {
            let text_data = match fs::read_to_string(file_path) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("❌ Failed to read text file {}: {}", file_path.display(), e);
                    continue;
                }
            };
            parse_text_ticks(&text_data, symbol)
        } else {
            let f = match File::open(file_path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("❌ Failed to open file {}: {}", file_path.display(), e);
                    continue;
                }
            };
            let metadata = match f.metadata() {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("❌ Failed to get metadata for {}: {}", file_path.display(), e);
                    continue;
                }
            };
            if metadata.len() == 0 {
                eprintln!("⚠️ Skipping empty file: {}", file_path.display());
                continue;
            }
            let mut compressed = Vec::new();
            if let Err(e) = BufReader::new(f).read_to_end(&mut compressed) {
                eprintln!("❌ Failed to read file {}: {}", file_path.display(), e);
                continue;
            }
            let mut decompressed = Vec::new();
            match lzma_decompress(&mut Cursor::new(compressed), &mut decompressed) {
                Ok(_) => parse_bi5(&decompressed, symbol),
                Err(e) => {
                    eprintln!("⚠️ LZMA decompress failed for {}: {}", file_path.display(), e);
                    continue;
                }
            }
        };

        println!("📊 Processed {} ticks from {} for {}", ticks.len(), file_name, symbol);

        if !ticks.is_empty() {
            aggregate_ticks_to_ohlcv(&ticks, &base_date, &timeframe, writer.as_mut().unwrap(), &schema);
            total_bars += ticks.len();
        }
    }

    if total_bars == 0 {
        eprintln!("⚠️ No OHLCV bars generated for {} (timeframe: {}). Check input data.", symbol, timeframe_str);
        if let Some(writer) = writer {
            drop(writer);
        }
        let _ = fs::remove_file(&output_path);
        return;
    }

    writer.take().unwrap().close().expect("Failed to close Parquet writer");
    println!("✅ Created training data: {} ({} ticks processed)", output_path.display(), total_bars);
}

fn main() {
    let root_dir = "./data";
    let output_dir = "./data/training";
    let symbols = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF"];
    let timeframes = vec![
        TimeFrame::Seconds(3)
    ];

    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    symbols.par_iter().for_each(|&symbol| {
        let symbol_dir = format!("{}/{}", root_dir, symbol);
        let files: Vec<_> = match fs::read_dir(&symbol_dir) {
            Ok(entries) => entries
                .filter_map(|entry| {
                    let path = entry.ok()?.path();
                    if path.is_file() && path.extension().map(|e| e == "bi5" || e == "txt").unwrap_or(false) {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect(),
            Err(e) => {
                eprintln!("⚠️ Failed to read directory {}: {}", symbol_dir, e);
                Vec::new()
            }
        };

        if files.is_empty() {
            eprintln!("⚠️ No .bi5 or .txt files found for {}", symbol);
            return;
        }

        let file_refs: Vec<&Path> = files.iter().map(|p| p.as_path()).collect();

        timeframes.par_iter().for_each(|timeframe| {
            println!("🔄 Processing {} with timeframe {:?}", symbol, timeframe);
            convert_to_training_data(&file_refs, symbol, output_dir, timeframe.clone());
        });
    });

    println!("🎯 Training data generation complete!");
    println!("📁 Check {} for your IBKR-compatible datasets", output_dir);
}
