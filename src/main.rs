use std::{fs::{self, File}, io::{BufReader, Cursor, Read}, path::Path, sync::Arc};
use byteorder::{BigEndian, ReadBytesExt};
use chrono::{NaiveDate, Duration, DateTime, Utc, TimeZone};
use lzma_rs::lzma_decompress;
use parquet::{
    file::properties::WriterProperties,
    arrow::ArrowWriter,
    basic::Compression,
};
use arrow::{
    record_batch::RecordBatch,
    array::{Float64Array, TimestampMillisecondArray},
    datatypes::{Schema, Field, DataType, TimeUnit},
};

#[derive(Debug)]
struct Tick {
    timestamp_ms: u32,
    bid: f64,
    ask: f64,
    volume: f64,
}

fn normalize_price(price: u32, symbol: &str) -> f64 {
    match symbol {
        "EURUSD" | "GBPUSD" | "USDCHF" => price as f64 / 100000.0, // e.g., 113439 -> 1.13439
        "USDJPY" => price as f64 / 1000.0, // e.g., 145123 -> 145.123
        _ => {
            eprintln!("⚠️ Unknown symbol {}, using default normalization (1e5)", symbol);
            price as f64 / 100000.0
        }
    }
}

fn parse_bi5(data: &[u8], symbol: &str) -> Vec<Tick> {
    let mut cursor = Cursor::new(data);
    let mut ticks = Vec::new();

    while let Ok(ts) = cursor.read_u32::<BigEndian>() {
        let bid_points = cursor.read_u32::<BigEndian>().unwrap_or(0);
        let ask_points = cursor.read_u32::<BigEndian>().unwrap_or(0);
        let bid_volume = cursor.read_f32::<BigEndian>().unwrap_or(0.0);
        let ask_volume = cursor.read_f32::<BigEndian>().unwrap_or(0.0);

        let bid = normalize_price(bid_points, symbol);
        let ask = normalize_price(ask_points, symbol);
        let volume = (bid_volume + ask_volume) as f64;

        ticks.push(Tick {
            timestamp_ms: ts,
            bid,
            ask,
            volume,
        });
    }

    ticks
}

fn format_timestamp(base_date: &str, ms: u32) -> DateTime<Utc> {
    let base = NaiveDate::parse_from_str(base_date, "%Y%m%d")
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let full_time = base + Duration::milliseconds(ms as i64);
    Utc.from_utc_datetime(&full_time)
}

fn extract_date_from_filename(file_name: &str) -> Option<String> {
    let parts: Vec<&str> = file_name.split('_').collect();
    if parts.len() >= 3 {
        Some(parts[1].to_string()) // e.g., "20250601"
    } else {
        None
    }
}

fn convert_file_to_parquet(file_path: &Path, symbol: &str, output_dir: &str) {
    let file_name = file_path.file_name().unwrap().to_string_lossy();
    let output_path = Path::new(output_dir).join(file_name.as_ref()).with_extension("parquet");

    // Skip if output Parquet file already exists
    if output_path.exists() {
        println!("⏭️ Skipping {}: Output file {} already exists", file_path.display(), output_path.display());
        return;
    }

    let base_date = extract_date_from_filename(&file_name).unwrap_or_else(|| "20000101".into());
    println!("📂 Processing file: {}", file_path.display());

    let Ok(f) = File::open(file_path) else {
        eprintln!("❌ Failed to open file: {}", file_path.display());
        return;
    };

    let mut compressed = Vec::new();
    println!("📖 Reading file: {}", file_path.display());
    if let Err(e) = BufReader::new(f).read_to_end(&mut compressed) {
        eprintln!("❌ Failed to read file: {} — {}", file_path.display(), e);
        return;
    }

    if compressed.is_empty() || compressed.len() < 10 {
        eprintln!("❌ Invalid or empty compressed data for {} (size: {} bytes)", file_path.display(), compressed.len());
        return;
    }

    let mut decompressed = Vec::new();
    println!("🔄 Decompressing file: {}", file_path.display());
    if let Err(e) = lzma_decompress(&mut Cursor::new(compressed), &mut decompressed) {
        eprintln!("❌ LZMA decompress failed for {}: {}", file_path.display(), e);
        return;
    }

    let ticks = parse_bi5(&decompressed, symbol);
    println!("📊 Processed {} ticks from {}", ticks.len(), file_name);

    // Create Parquet schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
        Field::new("bid", DataType::Float64, false),
        Field::new("ask", DataType::Float64, false),
        Field::new("tick_volume", DataType::Float64, false),
    ]));

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("Failed to create output directory");
    }

    let file = File::create(&output_path).expect("Failed to create Parquet file");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).expect("Failed to create Parquet writer");

    // Convert ticks to Arrow arrays with explicit UTC timezone
    let timestamps: Vec<i64> = ticks.iter().map(|tick| format_timestamp(&base_date, tick.timestamp_ms).timestamp_millis()).collect();
    let bids: Vec<f64> = ticks.iter().map(|tick| tick.bid).collect();
    let asks: Vec<f64> = ticks.iter().map(|tick| tick.ask).collect();
    let volumes: Vec<f64> = ticks.iter().map(|tick| tick.volume).collect();

    // Create TimestampMillisecondArray with UTC timezone
    let timestamp_array = TimestampMillisecondArray::from(timestamps)
        .with_timezone_opt(Some("UTC".to_string()));

    // Create record batch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(timestamp_array),
            Arc::new(Float64Array::from(bids)),
            Arc::new(Float64Array::from(asks)),
            Arc::new(Float64Array::from(volumes)),
        ],
    ).expect("Failed to create record batch");

    writer.write(&batch).expect("Failed to write record batch");
    writer.close().expect("Failed to close Parquet writer");
    println!("✅ Converted {} → {}", file_name, output_path.display());
}

fn main() {
    let root_dir = "./data/USDCHF"; // Custom input directory
    let output_dir = "./data/output/USDCHF"; // Custom output directory
    if !Path::new(root_dir).exists() {
        eprintln!("❌ Input directory {} does not exist.", root_dir);
        return;
    }
    println!("📂 Scanning input directory: {}", root_dir);
    let symbols = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF"];

    let bi5_files: Vec<_> = fs::read_dir(root_dir)
        .expect("Failed to read input directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.is_file() && path.extension().map(|e| e == "bi5").unwrap_or(false) {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if bi5_files.is_empty() {
        eprintln!("⚠️ No .bi5 files found in {}", root_dir);
        return;
    }

    println!("📄 Found {} .bi5 files: {:?}", bi5_files.len(), bi5_files.iter().map(|p| p.file_name().unwrap().to_string_lossy().to_string()).collect::<Vec<_>>());

    for path in bi5_files {
        let file_name = path.file_name().unwrap().to_string_lossy();
        let symbol = symbols.iter().find(|&&s| file_name.starts_with(s));
        if let Some(symbol) = symbol {
            convert_file_to_parquet(&path, symbol, output_dir);
        } else {
            eprintln!("⚠️ Skipping file {}: Unknown symbol", file_name);
        }
    }
}
