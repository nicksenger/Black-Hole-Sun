//! Prompt-preparation operations for spec datatypes.
//!
//! `black-hole-type` is intentionally datatypes only; behavior such as token
//! construction and shaping sequences for prompting (trimming, padding,
//! framing) lives here instead.

use black_hole_type::{DarkToken, InferenceOutput, LogitEntry, SequenceOutput, IM_END, PAD};

/// Operations for constructing dark tokens.
pub trait TokenOps: Sized {
    /// Creates a dark token with a single one-hot logit entry at `token_id`.
    fn one_hot(token_id: u32) -> Self;
}

impl TokenOps for DarkToken {
    fn one_hot(token_id: u32) -> Self {
        Self {
            predicted: token_id,
            dark_knowledge: vec![LogitEntry {
                token_id,
                log_prob: 0.0,
            }],
        }
    }
}

/// Operations that shape a single output sequence for use as prompt input.
pub trait SeqPromptOps: Sized {
    /// Removes trailing pad tokens (PAD, IM_END) from the sequence.
    fn trim_padding(&mut self);

    /// Pads the sequence from the start to the specified length
    fn pad_start_to(&mut self, len: usize);
}

impl SeqPromptOps for SequenceOutput {
    fn trim_padding(&mut self) {
        if let Some(last_non_padding) = self
            .0
            .iter()
            .rposition(|dt| dt.predicted != PAD && dt.predicted != IM_END)
        {
            self.0.truncate(last_non_padding + 1);
        } else {
            self.0.clear();
        }
    }

    fn pad_start_to(&mut self, len: usize) {
        let mut pad = vec![DarkToken::one_hot(PAD); len.saturating_sub(self.0.len())];
        pad.append(&mut self.0);
        self.0 = pad;
    }
}

/// Operations that shape a batched inference output for use as prompt input.
pub trait InferPromptOps: Sized {
    /// Returns the total number of DarkTokens across all contained sequences.
    fn n_tokens(&self) -> usize;

    /// Pads all sequences from the start to the maximum sequence length.
    fn pad_start(&mut self);

    /// Removes trailing pad tokens from all sequences.
    fn trim_padding(&mut self);

    /// Trims padding from sequences, then frames them with the provided tokens
    /// and pads from start to the max length
    fn frame(&mut self, before: Vec<DarkToken>, after: Vec<DarkToken>);

    /// Frames each sequence with the corresponding before/after sequence from the provided iterator
    fn frame_with<T: Iterator<Item = (Vec<DarkToken>, Vec<DarkToken>)>>(&mut self, frames: T);
}

impl InferPromptOps for InferenceOutput {
    fn n_tokens(&self) -> usize {
        self.results.iter().map(|seq| seq.0.len()).sum()
    }

    fn pad_start(&mut self) {
        let max = self
            .results
            .iter()
            .map(|seq| seq.0.len())
            .max()
            .unwrap_or_default();
        for seq in &mut self.results {
            seq.pad_start_to(max);
        }
    }

    fn trim_padding(&mut self) {
        for seq in &mut self.results {
            seq.trim_padding();
        }
    }

    fn frame(&mut self, before: Vec<DarkToken>, after: Vec<DarkToken>) {
        self.trim_padding();
        for seq in &mut self.results {
            let mut new = before.clone();
            new.append(&mut seq.0);
            new.append(&mut after.clone());
            seq.0 = new;
        }
        self.pad_start();
    }

    fn frame_with<T: Iterator<Item = (Vec<DarkToken>, Vec<DarkToken>)>>(&mut self, frames: T) {
        self.trim_padding();
        for (seq, (before, mut after)) in &mut self.results.iter_mut().zip(frames) {
            let mut new = before;
            new.append(&mut seq.0);
            new.append(&mut after);
            seq.0 = new;
        }
        self.pad_start();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn tok(token_id: u32) -> DarkToken {
        DarkToken {
            predicted: token_id,
            dark_knowledge: vec![LogitEntry {
                token_id,
                log_prob: 0.0,
            }],
        }
    }

    #[test]
    fn test_trim() {
        let mut seq = SequenceOutput(
            [1, 2, 3, 4, 5, 248046, 248044, 248044, 248044]
                .into_iter()
                .map(tok)
                .collect(),
        );
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );

        let mut seq = SequenceOutput(
            [1, 2, 3, 4, 5, 248046, 248044, 248046, 248044]
                .into_iter()
                .map(tok)
                .collect(),
        );
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );

        let mut seq = SequenceOutput(
            [1, 2, 3, 4, 5, 248044, 248044, 248044, 248044]
                .into_iter()
                .map(tok)
                .collect(),
        );
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );

        let mut seq = SequenceOutput([1, 2, 3, 4, 5].into_iter().map(tok).collect());
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );

        let mut seq = SequenceOutput([].into_iter().map(tok).collect());
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            Vec::<u32>::new()
        );

        let mut seq = SequenceOutput([248044, 248044, 248044].into_iter().map(tok).collect());
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            Vec::<u32>::new()
        );

        let mut seq = SequenceOutput([248044, 248046].into_iter().map(tok).collect());
        seq.trim_padding();
        assert_eq!(
            seq.0.into_iter().map(|dt| dt.predicted).collect::<Vec<_>>(),
            Vec::<u32>::new()
        );

        let mut seq = SequenceOutput(vec![
            DarkToken {
                predicted: 248044,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248044,
                    log_prob: 0.0,
                }],
            },
            DarkToken {
                predicted: 248046,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248046,
                    log_prob: 0.4,
                }],
            },
            DarkToken {
                predicted: 248044,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248044,
                    log_prob: 0.0,
                }],
            },
            DarkToken {
                predicted: 248046,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248046,
                    log_prob: 0.8,
                }],
            },
        ]);
        seq.trim_padding();
        assert_eq!(
            seq.0
                .into_iter()
                .map(|dt| dt.dark_knowledge[0].log_prob)
                .collect::<Vec<_>>(),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn test_frame() {
        let mut out = InferenceOutput {
            results: vec![
                SequenceOutput(
                    [1, 2, 3, 4, 5, 248044, 248044, 248044, 248044]
                        .into_iter()
                        .map(tok)
                        .collect(),
                ),
                SequenceOutput(
                    [1, 2, 3, 248046, 248044, 248044, 248044]
                        .into_iter()
                        .map(tok)
                        .collect(),
                ),
                SequenceOutput([1, 248044, 248044].into_iter().map(tok).collect()),
            ],
        };
        out.frame(
            vec![DarkToken {
                predicted: 248045,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248045,
                    log_prob: 0.0,
                }],
            }],
            vec![DarkToken {
                predicted: 248046,
                dark_knowledge: vec![LogitEntry {
                    token_id: 248046,
                    log_prob: 0.0,
                }],
            }],
        );

        assert_eq!(
            out.results[0]
                .0
                .iter()
                .map(|dt| dt.predicted)
                .collect::<Vec<_>>(),
            vec![248045, 1, 2, 3, 4, 5, 248046]
        );
        assert_eq!(
            out.results[1]
                .0
                .iter()
                .map(|dt| dt.predicted)
                .collect::<Vec<_>>(),
            vec![248044, 248044, 248045, 1, 2, 3, 248046]
        );
        assert_eq!(
            out.results[2]
                .0
                .iter()
                .map(|dt| dt.predicted)
                .collect::<Vec<_>>(),
            vec![248044, 248044, 248044, 248044, 248045, 1, 248046]
        );
    }

    #[test]
    fn test_n_tokens() {
        let out = InferenceOutput {
            results: vec![
                SequenceOutput([1, 2, 3].into_iter().map(tok).collect()),
                SequenceOutput([4].into_iter().map(tok).collect()),
                SequenceOutput([5, 6].into_iter().map(tok).collect()),
            ],
        };

        assert_eq!(out.n_tokens(), 6);
    }
}
