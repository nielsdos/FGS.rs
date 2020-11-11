use crate::nucleotide::{
    convert_dna_nrs_to_letters, protein, trinucleotide, NUCL_A, NUCL_C, NUCL_G, NUCL_T,
    NUCL_UNKNOWN,
};
use crate::output_entry::OutputEntry;
use crate::run::InputConfig;
use crate::train::{TrainSingle, TrainStartStopSingle};
use crate::viterbi::{PathMatrix, ScoreMatrix, State};
use bio::alphabets::dna::revcomp;
use std::mem::replace;
use strum::IntoEnumIterator;

/// Strand type
#[derive(Debug, Eq, PartialEq)]
enum StrandType {
    Unknown,
    Positive,
    Negative,
}

/// Viterbi state after the algorithm has ran.
pub struct CalculatedViterbi<'a> {
    /// The input sequence.
    seq: &'a [u8],
    /// The training data specialised for this CG-area.
    train: &'a TrainSingle,
    /// The configuration for all processing inputs.
    input_cfg: InputConfig,
    /// The scoring matrix.
    alpha: ScoreMatrix,
    /// The reconstructed path matrix.
    path: PathMatrix,
}

impl<'a> CalculatedViterbi<'a> {
    /// New instance for a calculated viterbi state.
    pub fn new(
        seq: &'a [u8],
        train: &'a TrainSingle,
        input_cfg: InputConfig,
        alpha: ScoreMatrix,
        path: PathMatrix,
    ) -> Self {
        Self {
            seq,
            train,
            input_cfg,
            alpha,
            path,
        }
    }

    /// Calculates refine offset of genome.
    fn refine(&self, pos: usize, train: &TrainStartStopSingle, reverse: bool) -> isize {
        // Fixup offset to be zero-based
        let pos = pos - 1;

        let is_not_stop_codon = |codon| {
            if reverse {
                codon != [NUCL_T, NUCL_T, NUCL_A]
                    && codon != [NUCL_C, NUCL_T, NUCL_A]
                    && codon != [NUCL_T, NUCL_C, NUCL_A]
            } else {
                codon != [NUCL_T, NUCL_A, NUCL_A]
                    && codon != [NUCL_T, NUCL_A, NUCL_G]
                    && codon != [NUCL_T, NUCL_G, NUCL_A]
            }
        };

        let is_start_codon = |codon| {
            if reverse {
                codon == [NUCL_C, NUCL_A, NUCL_T]
                    || codon == [NUCL_C, NUCL_A, NUCL_C]
                    || codon == [NUCL_C, NUCL_A, NUCL_A]
            } else {
                codon == [NUCL_A, NUCL_T, NUCL_G]
                    || codon == [NUCL_G, NUCL_T, NUCL_G]
                    || codon == [NUCL_T, NUCL_T, NUCL_G]
            }
        };

        let execute = |orf_inv_off: usize,
                       iterator: Box<dyn Iterator<Item = (usize, &'a [u8])> + 'a>|
         -> isize {
            let mut e_save = f64::MAX;
            let mut s_save = 0isize;

            // Find the optimal start codon within 30bp up- and downstream of start codon
            for (off, _) in iterator
                .take_while(|(off, codon)| {
                    (35..=self.seq.len() - 35).contains(off) && is_not_stop_codon(*codon)
                })
                .filter(|(_, codon)| is_start_codon(*codon))
            {
                let freq_sum = self.seq[off - orf_inv_off - 30..off - orf_inv_off + 30]
                    .windows(3)
                    .zip(&train[..])
                    .fold(0.0f64, |acc, (curr, array)| {
                        acc - array[trinucleotide(curr[0], curr[1], curr[2])]
                    });

                if freq_sum < e_save {
                    e_save = freq_sum;
                    s_save = (off - pos) as _;
                }
            }

            s_save
        };

        if reverse {
            execute(
                2,
                Box::new(
                    self.seq[pos - 2..]
                        .chunks_exact(3)
                        .enumerate()
                        .map(|(off, codon)| (pos + off * 3, codon)),
                ),
            )
        } else {
            // We need to start with pos % 3 to end up with full chunks.
            execute(
                0,
                Box::new(
                    self.seq[pos % 3..=pos + 2]
                        .chunks_exact(3)
                        .rev()
                        .enumerate()
                        .map(|(off, codon)| (pos - off * 3, codon)),
                ),
            )
        }
    }

    /// Backtrack the optimal path.
    pub fn backtrack(&self) -> Vec<OutputEntry> {
        let (refine, gene_len) = if self.input_cfg.whole_genome {
            (true, 120)
        } else {
            (false, 60)
        };

        let mut vpath = vec![State::S; self.seq.len()].into_boxed_slice();

        // Find the state for seq[n] with the highest probability
        let mut prob = f64::MAX;
        for i in State::iter() {
            let curr = self.alpha[self.seq.len() - 1][i as usize];
            if curr < prob {
                prob = curr;
                vpath[self.seq.len() - 1] = i;
            }
        }

        // Backtrack the optimal path
        for t in (0..=(self.seq.len() - 2)).rev() {
            vpath[t] = self.path[t + 1][vpath[t + 1] as usize].unwrap_or(State::S);
        }

        let mut strand_type = StrandType::Unknown;
        let mut start_t = 0;
        let mut dna_start_t_withstop = 0;
        let mut dna_start_t = 0;

        // Output
        let mut dna = Vec::new();
        let mut outputs = Vec::new();
        let mut insertions = Vec::new();
        let mut deletions = Vec::new();

        let mut start_orf = 0;
        let mut prev_match = 0;

        for (t, &vpath_t) in vpath.iter().enumerate() {
            if strand_type == StrandType::Unknown
                && start_t == 0
                && ((vpath_t >= State::M1 && vpath_t <= State::M6)
                    || (vpath_t >= State::M1_1 && vpath_t <= State::M6_1)
                    || vpath_t == State::S
                    || vpath_t == State::S_1)
            {
                start_t = t + 1;
                dna_start_t = start_t;
                dna_start_t_withstop = start_t;
            }

            if strand_type == StrandType::Unknown
                && (vpath_t == State::M1
                    || vpath_t == State::M4
                    || vpath_t == State::M1_1
                    || vpath_t == State::M4_1)
            {
                dna.clear();
                dna.push(self.seq[t]);
                dna_start_t = t + 1;

                dna_start_t_withstop =
                    if t > 2 && (vpath_t == State::M1_1 || vpath_t == State::M4_1) {
                        t - 2
                    } else {
                        dna_start_t
                    };

                start_orf = t + 1;
                prev_match = vpath_t as usize;

                // Determine if positive or negative strand.
                strand_type = if vpath_t < State::M6 {
                    StrandType::Positive
                } else {
                    StrandType::Negative
                };
            } else if strand_type != StrandType::Unknown
                && (vpath_t == State::E || vpath_t == State::E_1 || t == self.seq.len() - 1)
            {
                let mut end_t = if vpath_t == State::E || vpath_t == State::E_1 {
                    t + 3
                } else {
                    // Remove incomplete codon
                    let mut temp_t = t;
                    while vpath[temp_t] != State::M1
                        && vpath[temp_t] != State::M4
                        && vpath[temp_t] != State::M1_1
                        && vpath[temp_t] != State::M4_1
                    {
                        dna.pop();
                        temp_t -= 1;
                    }

                    t + 1
                };

                // Determine output.
                if dna.len() > gene_len {
                    let final_score = (self.alpha[end_t - 4][vpath[end_t - 4]]
                        - self.alpha[start_t + 2][vpath[start_t + 2]])
                        / (end_t as f64 - start_t as f64 - 5.0);

                    let frame = start_orf % 3;
                    let frame = if frame == 0 { 3 } else { frame };

                    // Handle the two cases: either a positive strand or a negative.
                    let (dna, protein, start) = if strand_type == StrandType::Positive {
                        if start_t + 3 == dna_start_t {
                            // Add complete start codon to dna, Ye April 21, 2016
                            debug_assert_eq!(dna.len() % 3, 0);
                            dna.insert(0, self.seq[dna_start_t + 1]);
                            dna.insert(0, self.seq[dna_start_t]);
                            dna.insert(0, self.seq[dna_start_t - 1]);
                            dna_start_t -= 3; // Fixes the output offset, and also the double start codon in potential refinement
                        }

                        // Add refinement of the start codons here, Ye, April 16, 2016
                        if refine {
                            let s_save = self.refine(dna_start_t, &self.train.start, false);
                            // Will just wrap over if exceeds the limit of the type
                            dna_start_t = (dna_start_t as isize + s_save) as _;

                            // Update dna
                            if s_save != 0 {
                                dna.splice(
                                    0..0,
                                    self.seq[dna_start_t - 1..end_t - 1].iter().cloned(),
                                );
                            }
                        }

                        let protein = protein(&dna, true, self.input_cfg.whole_genome);

                        // Only process it if it's asked for.
                        let dna = if self.input_cfg.save_dna {
                            let mut dna = dna.clone();
                            convert_dna_nrs_to_letters(&mut dna);
                            Some(dna)
                        } else {
                            None
                        };

                        (dna, protein, dna_start_t)
                    } else {
                        // we know that we're in a negative strand.

                        if dna_start_t + dna.len() == end_t - 2 {
                            // Add complete start codon (on reverse strand) to dna, Ye April 21, 2016
                            debug_assert_eq!(dna.len() % 3, 0);
                            dna.push(self.seq[end_t - 3]);
                            dna.push(self.seq[end_t - 2]);
                            dna.push(self.seq[end_t - 1]);
                            end_t += 3; // Fixes the output offset, and also the double start codon in potential refinement
                        }

                        // Add refinement of the start codons here, Ye, April 16, 2016
                        if refine {
                            let s_save = self.refine(end_t, &self.train.stop1, true);
                            // Will just wrap over if exceeds the limit of the type
                            let new_end = (end_t as isize + s_save) as _;

                            // Update dna
                            if s_save != 0 {
                                dna.extend_from_slice(&self.seq[end_t - 1 - 2..new_end - 1]);
                            }

                            end_t = new_end;
                        }

                        let protein = protein(&dna, false, self.input_cfg.whole_genome);

                        // Only process it if it's asked for.
                        let dna = if self.input_cfg.save_dna {
                            convert_dna_nrs_to_letters(&mut dna);
                            Some(revcomp(&dna))
                        } else {
                            None
                        };

                        (dna, protein, dna_start_t_withstop)
                    };

                    // Add it to the list of outputs for this input.
                    outputs.push(OutputEntry::new(
                        dna,
                        protein,
                        start,
                        end_t,
                        final_score,
                        frame,
                        strand_type == StrandType::Positive,
                        replace(&mut insertions, Vec::new()),
                        replace(&mut deletions, Vec::new()),
                    ));
                }

                strand_type = StrandType::Unknown;
                start_t = 0;
                dna.clear();
            } else if strand_type != StrandType::Unknown
                && ((vpath_t >= State::M1 && vpath_t <= State::M6)
                    || (vpath_t >= State::M1_1 && vpath_t <= State::M6_1))
                && (vpath_t as usize) < 6 + prev_match
            {
                let out_nt = (vpath_t as usize)
                    + (if (vpath_t as usize) < prev_match {
                        6
                    } else {
                        0
                    })
                    - prev_match
                    - 1;

                for _ in 0..out_nt {
                    dna.push(NUCL_UNKNOWN);
                    if self.input_cfg.save_meta {
                        deletions.push(t + 1);
                    }
                }

                dna.push(self.seq[t]);
                prev_match = vpath_t as usize;
            } else if self.input_cfg.save_meta
                && strand_type != StrandType::Unknown
                && ((vpath_t >= State::I1 && vpath_t <= State::I6)
                    || (vpath_t >= State::I1_1 && vpath_t <= State::I6_1))
            {
                // Save insertions.
                insertions.push(t + 1);
            } else if strand_type != StrandType::Unknown && vpath_t == State::R {
                // For long NNNNNNNNN, pretend R state.
                strand_type = StrandType::Unknown;
                start_t = 0;
                dna.clear();
            }
        }

        outputs
    }
}
