//! Stanford Dogs dataset plumbing shared by the corgi toys, plus the
//! Hugging Face model/cache helpers they all repeat.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use black_hole_sun::{ArtifactDelivery, TensorContract, VoidOps};
use candle_datasets::hub::from_hub;
use hf_hub::HFClientSync;
use parquet::record::{Field, Row};
use rand::seq::{IndexedRandom, SliceRandom};
use serde::{Deserialize, Serialize};

/// Target image size for the ResNet-18 toys.
pub const IMAGE_SIZE: usize = 224;
/// Hugging Face dataset with the Stanford Dogs images and labels.
pub const DATASET_ID: &str = "maurice-fp/stanford-dogs";
/// Total number of samples in the dataset.
pub const DATASET_SAMPLES: usize = 20_580;

const PEMBROKE_LABEL: u32 = 111;
const CARDIGAN_LABEL: u32 = 112;

/// Metadata attached to every corgi input emission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleMetadata {
    pub dataset_label: u32,
}

/// One decoded Stanford Dogs sample.
#[derive(Debug)]
pub struct DatasetSample {
    image: Vec<u8>,
    label: u32,
}

struct DatasetCursor {
    samples: std::vec::IntoIter<DatasetSample>,
}

impl DatasetCursor {
    fn new() -> Result<Self, String> {
        let api = HFClientSync::new().map_err(|e| format!("Hugging Face API: {e}"))?;

        // Select positions before loading images so balancing does not require
        // keeping the complete dataset in memory. All corgis are retained,
        // while the non-corgi side is an equally sized random sample.
        let mut corgi_positions = Vec::new();
        let mut other_positions = Vec::new();
        for (file_index, reader) in dataset_readers(&api)?.into_iter().enumerate() {
            for (row_index, row) in reader.into_iter().enumerate() {
                let row = row.map_err(|error| format!("read Stanford Dogs row: {error}"))?;
                let position = (file_index, row_index);
                if is_corgi(parse_label(&row)?) {
                    corgi_positions.push(position);
                } else {
                    other_positions.push(position);
                }
            }
        }
        if corgi_positions.is_empty() {
            return Err("Stanford Dogs dataset has no corgi samples".to_owned());
        }
        let mut rng = rand::rng();
        let selected = select_balanced_positions(corgi_positions, other_positions, &mut rng)?;
        let expected_samples = selected.len();

        let mut samples = Vec::with_capacity(expected_samples);
        for (file_index, reader) in dataset_readers(&api)?.into_iter().enumerate() {
            for (row_index, row) in reader.into_iter().enumerate() {
                let row = row.map_err(|error| format!("read Stanford Dogs row: {error}"))?;
                if selected.contains(&(file_index, row_index)) {
                    samples.push(parse_row(row)?);
                }
            }
        }
        samples.shuffle(&mut rng);
        if samples.len() != expected_samples {
            return Err(format!(
                "selected {expected_samples} Stanford Dogs samples, but loaded {}",
                samples.len()
            ));
        }

        Ok(Self {
            samples: samples.into_iter(),
        })
    }

    fn next(&mut self) -> Result<Option<DatasetSample>, String> {
        Ok(self.samples.next())
    }
}

fn dataset_readers(
    api: &HFClientSync,
) -> Result<Vec<parquet::file::serialized_reader::SerializedFileReader<std::fs::File>>, String> {
    from_hub(api, DATASET_ID.to_owned())
        .map_err(|error| format!("download Stanford Dogs parquet: {error}"))
}

fn select_balanced_positions(
    corgi_positions: Vec<(usize, usize)>,
    other_positions: Vec<(usize, usize)>,
    rng: &mut impl rand::Rng,
) -> Result<HashSet<(usize, usize)>, String> {
    let corgi_count = corgi_positions.len();
    if other_positions.len() < corgi_count {
        return Err(format!(
            "Stanford Dogs dataset has only {} non-corgi samples for {} corgis",
            other_positions.len(),
            corgi_count
        ));
    }

    let mut selected = corgi_positions;
    selected.extend(other_positions.sample(rng, corgi_count).copied());
    Ok(selected.into_iter().collect())
}

static DATASET: OnceLock<Result<Mutex<DatasetCursor>, String>> = OnceLock::new();

/// The next dataset sample, shared across all callers in this process.
pub fn next_sample() -> Result<DatasetSample, String> {
    let cursor = DATASET
        .get_or_init(|| DatasetCursor::new().map(Mutex::new))
        .as_ref()
        .map_err(Clone::clone)?;
    cursor
        .lock()
        .map_err(|_| "Stanford Dogs cursor poisoned".to_owned())?
        .next()?
        .ok_or_else(|| "Stanford Dogs dataset exhausted".to_owned())
}

fn parse_row(row: Row) -> Result<DatasetSample, String> {
    let mut image = None;
    let label = parse_label(&row)?;
    for (_name, field) in row.get_column_iter() {
        match field {
            Field::Group(group) => {
                for (name, nested) in group.get_column_iter() {
                    if name == "bytes" {
                        if let Field::Bytes(bytes) = nested {
                            image = Some(bytes.data().to_vec());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(DatasetSample {
        image: image.ok_or_else(|| "dataset row has no image bytes".to_owned())?,
        label,
    })
}

fn parse_label(row: &Row) -> Result<u32, String> {
    row.get_column_iter()
        .find_map(|(_name, field)| match field {
            Field::Long(value) => Some(*value as u32),
            _ => None,
        })
        .ok_or_else(|| "dataset row has no label".to_owned())
}

fn is_corgi(label: u32) -> bool {
    matches!(label, PEMBROKE_LABEL | CARDIGAN_LABEL)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_balanced_positions_keeps_all_corgis() {
        let corgis = vec![(0, 1), (0, 4), (1, 2)];
        let others = (0..10).map(|row| (2, row)).collect();
        let mut rng = rand::rng();

        let selected = select_balanced_positions(corgis.clone(), others, &mut rng).unwrap();

        assert_eq!(selected.len(), corgis.len() * 2);
        assert!(corgis
            .into_iter()
            .all(|position| selected.contains(&position)));
        assert_eq!(selected.iter().filter(|(file, _)| *file == 2).count(), 3);
    }

    #[test]
    fn select_balanced_positions_rejects_too_few_other_samples() {
        let result =
            select_balanced_positions(vec![(0, 0), (0, 1)], vec![(1, 0)], &mut rand::rng());

        assert!(result.is_err());
    }
}

/// Decode one dataset image into a normalized CHW f32 vector.
pub fn image_tensor(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let image = image::load_from_memory(bytes)
        .map_err(|error| format!("decode dataset image: {error}"))?
        .resize_to_fill(
            IMAGE_SIZE as u32,
            IMAGE_SIZE as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();
    let mut output = vec![0.0; 3 * IMAGE_SIZE * IMAGE_SIZE];
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for (index, pixel) in image.pixels().enumerate() {
        let y = index / IMAGE_SIZE;
        let x = index % IMAGE_SIZE;
        for channel in 0..3 {
            output[channel * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] =
                (f32::from(pixel[channel]) / 255.0 - mean[channel]) / std[channel];
        }
    }
    Ok(output)
}

/// Generate one normalized Stanford Dogs image as a typed input emission for
/// contract `C`. Shared by the forward-only, backward, and two-sided ZO
/// examples so all of them exercise the same dataset and input contract.
pub async fn generate_image<J, C>(jungle: &J) -> Result<ArtifactDelivery<C::Input>, String>
where
    J: VoidOps,
    C: TensorContract<Metadata = SampleMetadata>,
{
    let sample = next_sample()?;
    let values = image_tensor(&sample.image)?;
    let tensor = C::input_f32(&[1, 3, IMAGE_SIZE, IMAGE_SIZE], values);
    let metadata = SampleMetadata {
        dataset_label: sample.label,
    };
    jungle.emit::<C>(&[tensor], &metadata).await
}

/// Resolve the ResNet-18 checkpoint path, downloading it from Hugging Face
/// when no local path is given.
pub fn model_path(argument: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(path) = argument {
        return Ok(path);
    }
    let api = HFClientSync::new().map_err(|error| format!("Hugging Face API: {error}"))?;
    api.model("lmz".to_owned(), "candle-resnet".to_owned())
        .download_file()
        .filename("resnet18.safetensors")
        .send()
        .map_err(|error| format!("download ResNet-18 checkpoint: {error}"))
}

/// Point `HF_HUB_CACHE` at a writable directory: an explicit argument, the
/// existing environment value, or `target/<default_subdir>/huggingface`.
pub fn configure_hf_cache(
    argument: Option<PathBuf>,
    default_subdir: &str,
) -> Result<PathBuf, String> {
    let local = std::env::current_dir()
        .map_err(|error| format!("current directory: {error}"))?
        .join(format!("target/{default_subdir}/huggingface"));
    let explicit = argument.is_some();
    let requested =
        argument.or_else(|| std::env::var_os("HF_HUB_CACHE").map(PathBuf::from));
    if let Some(path) = requested {
        if writable(&path).is_ok() {
            std::env::set_var("HF_HUB_CACHE", &path);
            return Ok(path);
        }
        if explicit {
            return Err(format!("Hugging Face cache is not writable: {}", path.display()));
        }
        eprintln!(
            "ignoring non-writable HF_HUB_CACHE {}; using {}",
            path.display(),
            local.display()
        );
    }
    writable(&local)
        .map_err(|error| format!("create Hugging Face cache: {error}"))?;
    std::env::set_var("HF_HUB_CACHE", &local);
    Ok(local)
}

fn writable(path: &PathBuf) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    let probe = path.join(".write-test");
    std::fs::write(&probe, [])?;
    std::fs::remove_file(probe)
}
