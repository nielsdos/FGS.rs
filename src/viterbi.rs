use crate::backtrack::CalculatedViterbi;
use crate::hmm::{Hmm, Transition};
use crate::matrix::{split_matrix, MatrixRefs};
use crate::nucleotide::{trinucleotide, NUCL_A, NUCL_C, NUCL_G, NUCL_T, NUCL_UNKNOWN};
use crate::output_entry::OutputEntry;
use crate::run::{InputConfig, MIN_ACCEPTED_LEN};
use crate::train::{
    Train, TrainDistSingle, TrainSingle, TrainStartStopSingle, TrainTransSingle, MAX_CG,
};
use crate::util::{clamp, clampf64};
use std::cmp::min;
use std::{f64, ops};
use strum::IntoEnumIterator;
use strum_macros::{EnumCount, EnumIter};

/// Possible states in the algorithm.
/// Note: keep order correct
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, EnumIter, EnumCount)]
#[allow(non_camel_case_types)]
pub enum State {
    S = 0,
    E = 1,
    R = 2,
    S_1 = 3,
    E_1 = 4,
    M1 = 5,
    M2 = 6,
    M3 = 7,
    M4 = 8,
    M5 = 9,
    M6 = 10,
    M1_1 = 11,
    M2_1 = 12,
    M3_1 = 13,
    M4_1 = 14,
    M5_1 = 15,
    M6_1 = 16,
    I1 = 17,
    _I2 = 18,
    _I3 = 19,
    _I4 = 20,
    _I5 = 21,
    I6 = 22,
    I1_1 = 23,
    _I2_1 = 24,
    _I3_1 = 25,
    _I4_1 = 26,
    _I5_1 = 27,
    I6_1 = 28,
}

pub type PathState = Option<State>;
pub type ScoreMatrix = Box<[[f64; STATE_COUNT]]>;
pub type PathMatrix = Box<[[PathState; STATE_COUNT]]>;

impl State {
    /// Iterate until another state.
    pub fn iter_until(self, until: Self) -> impl DoubleEndedIterator<Item = State> {
        self.iter_until_usize(until as usize - self as usize)
    }

    /// Iterate until state+until is reached (inclusive). It's basically a convenient "take iterator".
    pub fn iter_until_usize(self, until: usize) -> impl DoubleEndedIterator<Item = State> {
        // Using strum with take & skip is rather slow.
        ((self as usize)..=(self as usize + until)).map(|n| Self::nth(n).unwrap())
    }

    /// Gets the nth enum variant.
    #[inline]
    pub fn nth(n: usize) -> Option<Self> {
        State::iter().get(n)
    }
}

impl<T> ops::Index<State> for [T] {
    type Output = T;

    fn index(&self, index: State) -> &Self::Output {
        &self[index as usize]
    }
}

impl<T> ops::IndexMut<State> for [T] {
    fn index_mut(&mut self, index: State) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

/// Viterbi state.
struct Viterbi<'a> {
    /// The Hmm: initial probability distribution + transition probabilities.
    hmm: &'a Hmm,
    /// The training data specialised for this CG-area.
    train: &'a TrainSingle,
    /// The input sequence.
    seq: &'a [u8],
    /// The configuration for all processing inputs.
    input_cfg: InputConfig,
}

impl<'a> Viterbi<'a> {
    /// New instance for the viterbi algorithm.
    pub fn new(
        hmm: &'a Hmm,
        train: &'a TrainSingle,
        seq: &'a [u8],
        input_cfg: InputConfig,
    ) -> Self {
        Self {
            hmm,
            train,
            seq,
            input_cfg,
        }
    }

    /// Calculate adjustment based on probability distribution for start and end' (which is start in minus strand).
    fn start_freq_adjustment_start_endminus(&self, t: usize, train: &TrainStartStopSingle) -> f64 {
        if t >= 30 {
            self.seq[t - 30..]
                .windows(3)
                .zip(&train[..])
                .fold(0.0f64, |acc, (curr, array)| {
                    acc - array[trinucleotide(curr[0], curr[1], curr[2])]
                })
        } else {
            self.seq
                .windows(3)
                .zip(&train[30 - t..])
                .fold(0.0f64, |acc, (curr, array)| {
                    acc - array[trinucleotide(curr[0], curr[1], curr[2])]
                })
                * 61.0
                / (30.0 + t as f64 + 1.0)
        }
    }

    /// Update inside the loop for M(') state
    fn update_inner_m(
        &mut self,
        i: State,
        alphas: &mut MatrixRefs<f64>,
        paths: &mut MatrixRefs<PathState>,
        from2: usize,
        to: usize,
        m1: State,
        train_trans: &TrainTransSingle,
    ) {
        let log_25 = (0.25f64).ln();

        // From M(') state
        let j = if i == m1 {
            m1 as usize + 5
        } else {
            i as usize - 1
        };
        let temp = -self.hmm.tr[Transition::MD]
            - train_trans[i as usize - m1 as usize][from2][to]
            - self.hmm.tr[Transition::DM];
        let mut best_alpha = alphas.t_1[j]
            - self.hmm.tr[Transition::MM]
            - train_trans[i as usize - m1 as usize][from2][to]
            - temp;
        if i == m1 {
            best_alpha -= self.hmm.tr[Transition::GG];
        }
        paths.t0[i] = State::nth(j);

        // From D state
        if !self.input_cfg.whole_genome {
            // Loop start var
            let start = if i == m1 { 4 } else { 5 };

            for j in m1.iter_until_usize(start) {
                let num_d = {
                    let j = j as i32;
                    let i = i as i32;
                    if j >= i {
                        i - j + 6
                    } else if j + 1 < i {
                        i - j
                    } else {
                        0
                    }
                };

                if num_d > 0 {
                    let num_d = num_d as f64;
                    let temp_alpha = alphas.t_1[j]
                        - log_25 * (num_d - 1.0)
                        - self.hmm.tr[Transition::DD] * (num_d - 2.0);
                    if temp_alpha < best_alpha {
                        best_alpha = temp_alpha;
                        paths.t0[i] = Some(j);
                    }
                }
            }
        }

        alphas.t0[i] = best_alpha + temp;
    }

    /// Update I(') state.
    fn update_i(
        &mut self,
        from: usize,
        to: usize,
        alphas: &mut MatrixRefs<f64>,
        paths: &mut MatrixRefs<PathState>,
        t: usize,
        i1: State,
        m1: State,
        temp_i: &mut [usize; 6],
        update_from_m: bool,
    ) {
        for i in i1.iter_until_usize(5) {
            // From I state
            alphas.t0[i] = alphas.t_1[i] - self.hmm.tr[Transition::II] - self.hmm.tr_i_i[from][to];
            paths.t0[i] = Some(i);

            // From M state
            if update_from_m {
                let j = i as usize - i1 as usize + m1 as usize;
                let mut temp_alpha =
                    alphas.t_1[j] - self.hmm.tr[Transition::MI] - self.hmm.tr_m_i[from][to];
                if i as usize == i1 as usize + 5 {
                    temp_alpha -= self.hmm.tr[Transition::GG];
                }
                if temp_alpha < alphas.t0[i] {
                    alphas.t0[i] = temp_alpha;
                    paths.t0[i] = State::nth(j);
                    temp_i[i as usize - i1 as usize] = t - 1;
                }
            }
        }
    }

    /// Adjusted the probability from a distribution.
    fn adjusted_probability_from_dist(&self, start_freq: f64, array: &TrainDistSingle) -> f64 {
        let h_kd = array[2] * (-(start_freq - array[1]).powi(2) / (2.0 * array[0].powi(2))).exp();
        let r_kd = array[5] * (-(start_freq - array[4]).powi(2) / (2.0 * array[3].powi(2))).exp();
        let p_kd = clampf64(h_kd / (h_kd + r_kd), 0.01, 0.99);
        p_kd.ln()
    }

    /// Run the algorithm to construct the score and path matrix.
    pub fn run(mut self) -> CalculatedViterbi<'a> {
        assert!(self.seq.len() >= MIN_ACCEPTED_LEN);

        let log_53 = (0.53f64).ln();
        let log_16 = (0.16f64).ln();
        let log_30 = (0.30f64).ln();
        let log_25 = (0.25f64).ln();
        let log_95 = (0.95f64).ln();
        let log_54 = (0.54f64).ln();
        let log_83 = (0.83f64).ln();
        let log_07 = (0.07f64).ln();
        let log_10 = (0.10f64).ln();

        // Holds at which position the most likely transition from a M to an I was.
        let mut insert_pos = [0usize; 6];
        // Same, but for accent state.
        let mut insert_pos_1 = [0usize; 6];

        let mut alpha = vec![[0.0f64; STATE_COUNT]; self.seq.len()].into_boxed_slice();
        let mut path = vec![[None; STATE_COUNT]; self.seq.len()].into_boxed_slice();

        // (already negated when reading)
        alpha[0].copy_from_slice(&self.hmm.pi[..]);

        // Stop state
        if self.seq[0..3] == [NUCL_T, NUCL_A, NUCL_A]
            || self.seq[0..3] == [NUCL_T, NUCL_A, NUCL_G]
            || self.seq[0..3] == [NUCL_T, NUCL_G, NUCL_A]
        {
            alpha[0][State::E] = f64::MAX;
            alpha[1][State::E] = f64::MAX;
            path[1][State::E] = Some(State::E);
            path[2][State::E] = Some(State::E);

            alpha[2][State::M6] = f64::MAX;
            alpha[1][State::M5] = f64::MAX;
            alpha[0][State::M4] = f64::MAX;
            alpha[2][State::M3] = f64::MAX;
            alpha[1][State::M2] = f64::MAX;
            alpha[0][State::M1] = f64::MAX;

            if self.seq[1..3] == [NUCL_A, NUCL_A] {
                alpha[2][State::E] -= log_53;
            } else if self.seq[1..3] == [NUCL_A, NUCL_G] {
                alpha[2][State::E] -= log_16;
            } else if self.seq[1..3] == [NUCL_G, NUCL_A] {
                alpha[2][State::E] -= log_30;
            }
        }

        // Start state
        if self.seq[0..3] == [NUCL_T, NUCL_T, NUCL_A]
            || self.seq[0..3] == [NUCL_C, NUCL_T, NUCL_A]
            || self.seq[0..3] == [NUCL_T, NUCL_C, NUCL_A]
        {
            alpha[0][State::S_1] = f64::MAX;
            alpha[1][State::S_1] = f64::MAX;
            alpha[2][State::S_1] = alpha[0][State::S];
            path[1][State::S_1] = Some(State::S_1);
            path[2][State::S_1] = Some(State::S_1);

            alpha[2][State::M6_1] = f64::MAX;
            alpha[2][State::M3_1] = f64::MAX;

            if self.seq[0..2] == [NUCL_T, NUCL_T] {
                alpha[2][State::S_1] -= log_53;
            } else if self.seq[0..2] == [NUCL_C, NUCL_T] {
                alpha[2][State::S_1] -= log_16;
            } else if self.seq[0..2] == [NUCL_T, NUCL_C] {
                alpha[2][State::S_1] -= log_30;
            }
        }

        // Fill out the rest of the columns
        let mut num_n = 0;
        for t in 1..self.seq.len() {
            // Update I' from M state?
            // Since FGS reads out of bounds, and thus the "!=" is always true, we need to emulate this.
            let update_i_accent_from_m = (t < 3 || path[t - 3][State::S_1] != Some(State::R))
                && (t < 4 || path[t - 4][State::S_1] != Some(State::R))
                && (t < 5 || path[t - 5][State::S_1] != Some(State::R));

            let mut alphas = split_matrix(&mut alpha, t - 1);
            let mut paths = split_matrix(&mut path, t - 1);

            // Note: if DNA is other than ACGT, do it later
            let from0 = if t >= 2 {
                let symbol0 = self.seq[t - 2];
                if symbol0 == NUCL_UNKNOWN {
                    NUCL_G
                } else {
                    symbol0
                }
            } else {
                NUCL_G
            } as usize;
            let symbol = self.seq[t - 1];
            let from = if symbol == NUCL_UNKNOWN {
                NUCL_G
            } else {
                symbol
            } as usize;
            let to = {
                let to_symbol = self.seq[t];
                if to_symbol == NUCL_UNKNOWN {
                    num_n += 1;
                    NUCL_G
                } else {
                    num_n = 0;
                    to_symbol
                }
            } as usize;
            let from2 = (from0 * 4 + from) & 15; // & 15 eliminates the bound check

            // M State
            for i in State::M1.iter_until(State::M6) {
                if alphas.t0[i] < f64::MAX {
                    self.update_inner_m(
                        i,
                        &mut alphas,
                        &mut paths,
                        from2,
                        to,
                        State::M1,
                        &self.train.trans,
                    );

                    // From start state
                    if i == State::M1 {
                        let temp_alpha = alphas.t_1[State::S] - self.train.trans[0][from2][to];
                        if temp_alpha < alphas.t0[State::M1] {
                            alphas.t0[State::M1] = temp_alpha;
                            paths.t0[State::M1] = Some(State::S);
                        }
                    }

                    // From I state
                    let j = if i == State::M1 {
                        State::I6 as usize
                    } else {
                        State::I1 as usize + (i as usize - State::M1 as usize - 1)
                    };

                    // To avoid stop codon
                    if t >= 2
                        && !((i == State::M2 || i == State::M5)
                            && t + 1 < self.seq.len()
                            && self.seq[insert_pos[j - State::I1 as usize]] == NUCL_T
                            && (self.seq[t..t + 2] == [NUCL_A, NUCL_A]
                                || self.seq[t..t + 2] == [NUCL_A, NUCL_G]
                                || self.seq[t..t + 2] == [NUCL_G, NUCL_A]))
                        && !((i == State::M3 || i == State::M6)
                            && insert_pos[j - State::I1 as usize] >= 1
                            && self.seq[insert_pos[j - State::I1 as usize] - 1] == NUCL_T
                            && ((self.seq[insert_pos[j - State::I1 as usize]] == NUCL_A
                                && self.seq[t] == NUCL_A)
                                || (self.seq[insert_pos[j - State::I1 as usize]] == NUCL_A
                                    && self.seq[t] == NUCL_G)
                                || (self.seq[insert_pos[j - State::I1 as usize]] == NUCL_G
                                    && self.seq[t] == NUCL_A)))
                    {
                        let temp_alpha = alphas.t_1[j] - self.hmm.tr[Transition::IM] - log_25;
                        if temp_alpha < alphas.t0[i] {
                            alphas.t0[i] = temp_alpha;
                            paths.t0[i] = State::nth(j);
                        }
                    }
                }
            }

            // I state
            self.update_i(
                from,
                to,
                &mut alphas,
                &mut paths,
                t,
                State::I1,
                State::M1,
                &mut insert_pos,
                true,
            );

            // M' state
            for i in State::M1_1.iter_until(State::M6_1) {
                if t >= 3
                    && (i == State::M1_1 || i == State::M4_1)
                    && (self.seq[t - 3..t] == [NUCL_T, NUCL_T, NUCL_A]
                        || self.seq[t - 3..t] == [NUCL_T, NUCL_C, NUCL_A]
                        || self.seq[t - 3..t] == [NUCL_C, NUCL_T, NUCL_A])
                {
                    // From Start state since this is actually stop codon in minus strand
                    alphas.t0[i] = alphas.t_1[State::S_1]
                        - self.train.rtrans[i as usize - State::M1_1 as usize][from2][to];
                    paths.t0[i] = Some(State::S_1);
                } else {
                    self.update_inner_m(
                        i,
                        &mut alphas,
                        &mut paths,
                        from2,
                        to,
                        State::M1_1,
                        &self.train.rtrans,
                    );

                    // From I state
                    let j = if i == State::M1_1 {
                        State::I6_1 as usize
                    } else {
                        State::I1_1 as usize + (i as usize - State::M1_1 as usize - 1)
                    };

                    // To avoid stop codon
                    if t >= 2
                        && !((i == State::M2_1 || i == State::M5_1)
                            && t + 1 < self.seq.len()
                            && self.seq[t + 1] == NUCL_A
                            && ((self.seq[insert_pos_1[j - State::I1_1 as usize]] == NUCL_T
                                && self.seq[t] == NUCL_T)
                                || (self.seq[insert_pos_1[j - State::I1_1 as usize]] == NUCL_C
                                    && self.seq[t] == NUCL_T)
                                || (self.seq[insert_pos_1[j - State::I1_1 as usize]] == NUCL_T
                                    && self.seq[t] == NUCL_C)))
                        && !((i == State::M3_1 || i == State::M6_1)
                            && self.seq[t] == NUCL_A
                            && insert_pos_1[j - State::I1_1 as usize] >= 1
                            && ((self.seq[insert_pos_1[j - State::I1_1 as usize] - 1] == NUCL_T
                                && self.seq[insert_pos_1[j - State::I1_1 as usize]] == NUCL_T)
                                || (self.seq[insert_pos_1[j - State::I1_1 as usize] - 1] == NUCL_C
                                    && self.seq[insert_pos_1[j - State::I1_1 as usize]] == NUCL_T)
                                || (self.seq[insert_pos_1[j - State::I1_1 as usize] - 1] == NUCL_T
                                    && self.seq[insert_pos_1[j - State::I1_1 as usize]] == NUCL_C)))
                    {
                        let temp_alpha = alphas.t_1[j] - self.hmm.tr[Transition::IM] - log_25;
                        if temp_alpha < alphas.t0[i] {
                            alphas.t0[i] = temp_alpha;
                            paths.t0[i] = State::nth(j);
                        }
                    }
                }
            }

            // I' state
            self.update_i(
                from,
                to,
                &mut alphas,
                &mut paths,
                t,
                State::I1_1,
                State::M1_1,
                &mut insert_pos_1,
                update_i_accent_from_m,
            );

            // Non coding state
            {
                alphas.t0[State::R] = alphas.t_1[State::R]
                    - self.train.noncoding[from][to]
                    - self.hmm.tr[Transition::RR];
                paths.t0[State::R] = Some(State::R);

                let temp_alpha = alphas.t_1[State::E] - self.hmm.tr[Transition::ER];
                if temp_alpha < alphas.t0[State::R] {
                    alphas.t0[State::R] = temp_alpha;
                    paths.t0[State::R] = Some(State::E);
                }

                let temp_alpha = alphas.t_1[State::E_1] - self.hmm.tr[Transition::ER];
                if temp_alpha < alphas.t0[State::R] {
                    alphas.t0[State::R] = temp_alpha;
                    paths.t0[State::R] = Some(State::E_1);
                }

                alphas.t0[State::R] -= log_95;
            }

            // End state
            if alphas.t0[State::E] == 0.0 {
                alphas.t0[State::E] = f64::MAX;
                paths.t0[State::E] = None;

                if t + 2 < self.seq.len()
                    && (self.seq[t..t + 3] == [NUCL_T, NUCL_A, NUCL_A]
                        || self.seq[t..t + 3] == [NUCL_T, NUCL_A, NUCL_G]
                        || self.seq[t..t + 3] == [NUCL_T, NUCL_G, NUCL_A])
                {
                    // These can't fail because of the check above.
                    let alpha_t2 = alphas.t2.as_mut().unwrap();
                    let alpha_t1 = alphas.t1.as_mut().unwrap();
                    let path_t2 = paths.t2.as_mut().unwrap();
                    let path_t1 = paths.t1.as_mut().unwrap();

                    alpha_t2[State::E] = f64::MAX;

                    // transition from frame4,frame5,and frame6
                    let temp_alpha = alphas.t_1[State::M6] - self.hmm.tr[Transition::GE];
                    if temp_alpha < alpha_t2[State::E] {
                        alpha_t2[State::E] = temp_alpha;
                        paths.t0[State::E] = Some(State::M6);
                    }

                    // transition from frame1,frame2,and frame3
                    let temp_alpha = alphas.t_1[State::M3] - self.hmm.tr[Transition::GE];
                    if temp_alpha < alpha_t2[State::E] {
                        alpha_t2[State::E] = temp_alpha;
                        paths.t0[State::E] = Some(State::M3);
                    }

                    alphas.t0[State::E] = f64::MAX;
                    alpha_t1[State::E] = f64::MAX;
                    path_t1[State::E] = Some(State::E);
                    path_t2[State::E] = Some(State::E);

                    alpha_t2[State::M6] = f64::MAX;
                    alpha_t1[State::M5] = f64::MAX;
                    alphas.t0[State::M4] = f64::MAX;
                    alpha_t2[State::M3] = f64::MAX;
                    alpha_t1[State::M2] = f64::MAX;
                    alphas.t0[State::M1] = f64::MAX;

                    if self.seq[t + 1..t + 3] == [NUCL_A, NUCL_A] {
                        alpha_t2[State::E] -= log_54;
                    } else if self.seq[t + 1..t + 3] == [NUCL_A, NUCL_G] {
                        alpha_t2[State::E] -= log_16;
                    } else if self.seq[t + 1..t + 3] == [NUCL_G, NUCL_A] {
                        alpha_t2[State::E] -= log_30;
                    }

                    // adjustment based on probability distribution
                    let start_freq = if t >= 60 {
                        self.seq[t - 60..t]
                            .windows(3)
                            .zip(&self.train.stop[..])
                            .fold(0.0f64, |acc, (curr, stop)| {
                                acc - stop[trinucleotide(curr[0], curr[1], curr[2])]
                            })
                    } else {
                        self.seq[0..t]
                            .windows(3)
                            .zip(&self.train.stop[60 - t..])
                            .fold(0.0f64, |acc, (curr, stop)| {
                                acc - stop[trinucleotide(curr[0], curr[1], curr[2])]
                            })
                            * 58.0
                            / (-3.0 + t as f64 + 1.0)
                    };

                    alpha_t2[State::E] -=
                        self.adjusted_probability_from_dist(start_freq, &self.train.e_dist);
                }
            }

            // Start' state
            if alphas.t0[State::S_1] == 0.0 {
                alphas.t0[State::S_1] = f64::MAX;
                paths.t0[State::S_1] = None;

                if t + 2 < self.seq.len()
                    && (self.seq[t..t + 3] == [NUCL_T, NUCL_T, NUCL_A]
                        || self.seq[t..t + 3] == [NUCL_C, NUCL_T, NUCL_A]
                        || self.seq[t..t + 3] == [NUCL_T, NUCL_C, NUCL_A])
                {
                    // These can't fail because of the check above.
                    let alpha_t2 = alphas.t2.as_mut().unwrap();
                    let alpha_t1 = alphas.t1.as_mut().unwrap();
                    let path_t2 = paths.t2.as_mut().unwrap();
                    let path_t1 = paths.t1.as_mut().unwrap();

                    alphas.t0[State::S_1] = f64::MAX;
                    alpha_t1[State::S_1] = f64::MAX;
                    alpha_t2[State::S_1] = alphas.t_1[State::R] - self.hmm.tr[Transition::RS];
                    paths.t0[State::S_1] = Some(State::R);
                    path_t1[State::S_1] = Some(State::S_1);
                    path_t2[State::S_1] = Some(State::S_1);

                    let temp_alpha = alphas.t_1[State::E_1] - self.hmm.tr[Transition::ES];
                    if temp_alpha < alpha_t2[State::S_1] {
                        alpha_t2[State::S_1] = temp_alpha;
                        paths.t0[State::S_1] = Some(State::E_1);
                    }

                    let temp_alpha = alphas.t_1[State::E] - self.hmm.tr[Transition::ES1];
                    if temp_alpha < alpha_t2[State::S_1] {
                        alpha_t2[State::S_1] = temp_alpha;
                        paths.t0[State::S_1] = Some(State::E);
                    }

                    alpha_t2[State::M3_1] = f64::MAX;
                    alpha_t2[State::M6_1] = f64::MAX;

                    if self.seq[t..t + 2] == [NUCL_T, NUCL_T] {
                        alpha_t2[State::S_1] -= log_54;
                    } else if self.seq[t..t + 2] == [NUCL_C, NUCL_T] {
                        alpha_t2[State::S_1] -= log_16;
                    } else if self.seq[t..t + 2] == [NUCL_T, NUCL_C] {
                        alpha_t2[State::S_1] -= log_30;
                    }

                    // adjustment based on probability distribution
                    let start = min(self.seq.len() - 2, t + 3);
                    let start_freq = self.seq[start..]
                        .windows(3)
                        .zip(&self.train.start1[..])
                        .fold(0.0f64, |acc, (curr, start)| {
                            acc - start[trinucleotide(curr[0], curr[1], curr[2])]
                        });

                    alpha_t2[State::S_1] -=
                        self.adjusted_probability_from_dist(start_freq, &self.train.s1_dist);
                }
            }

            // Start state
            if alphas.t0[State::S] == 0.0 {
                alphas.t0[State::S] = f64::MAX;
                paths.t0[State::S] = None;

                if t + 2 < self.seq.len()
                    && (self.seq[t..t + 3] == [NUCL_A, NUCL_T, NUCL_G]
                        || self.seq[t..t + 3] == [NUCL_G, NUCL_T, NUCL_G]
                        || self.seq[t..t + 3] == [NUCL_T, NUCL_T, NUCL_G])
                {
                    // These can't fail because of the check above.
                    let alpha_t2 = alphas.t2.as_mut().unwrap();
                    let alpha_t1 = alphas.t1.as_mut().unwrap();
                    let path_t2 = paths.t2.as_mut().unwrap();
                    let path_t1 = paths.t1.as_mut().unwrap();

                    alphas.t0[State::S] = f64::MAX;
                    alpha_t1[State::S] = f64::MAX;
                    alpha_t2[State::S] = alphas.t_1[State::R] - self.hmm.tr[Transition::RS];
                    paths.t0[State::S] = Some(State::R);
                    path_t1[State::S] = Some(State::S);
                    path_t2[State::S] = Some(State::S);

                    let temp_alpha = alphas.t_1[State::E] - self.hmm.tr[Transition::ES];
                    if temp_alpha < alpha_t2[State::S] {
                        alpha_t2[State::S] = temp_alpha;
                        paths.t0[State::S] = Some(State::E);
                    }

                    let temp_alpha = alphas.t_1[State::E_1] - self.hmm.tr[Transition::ES1];
                    if temp_alpha < alpha_t2[State::S] {
                        alpha_t2[State::S] = temp_alpha;
                        paths.t0[State::S] = Some(State::E_1);
                    }

                    if self.seq[t] == NUCL_A {
                        alpha_t2[State::S] -= log_83;
                    } else if self.seq[t] == NUCL_G {
                        alpha_t2[State::S] -= log_10;
                    } else if self.seq[t] == NUCL_T {
                        alpha_t2[State::S] -= log_07;
                    }

                    let start_freq =
                        self.start_freq_adjustment_start_endminus(t, &self.train.start);
                    alpha_t2[State::S] -=
                        self.adjusted_probability_from_dist(start_freq, &self.train.s_dist);
                }
            }

            // End' state
            if alphas.t0[State::E_1] == 0.0 {
                alphas.t0[State::E_1] = f64::MAX;
                paths.t0[State::E_1] = None;

                if t < self.seq.len() - 2
                    && (self.seq[t..t + 3] == [NUCL_C, NUCL_A, NUCL_T]
                        || self.seq[t..t + 3] == [NUCL_C, NUCL_A, NUCL_C]
                        || self.seq[t..t + 3] == [NUCL_C, NUCL_A, NUCL_A])
                {
                    // These can't fail because of the check above.
                    let alpha_t2 = alphas.t2.as_mut().unwrap();
                    let alpha_t1 = alphas.t1.as_mut().unwrap();
                    let path_t2 = paths.t2.as_mut().unwrap();
                    let path_t1 = paths.t1.as_mut().unwrap();

                    alphas.t0[State::E_1] = f64::MAX;
                    alpha_t1[State::E_1] = f64::MAX;
                    alpha_t2[State::E_1] = alphas.t_1[State::M6_1] - self.hmm.tr[Transition::GE];
                    paths.t0[State::E_1] = Some(State::M6_1);
                    path_t1[State::E_1] = Some(State::E_1);
                    path_t2[State::E_1] = Some(State::E_1);

                    if self.seq[t + 2] == NUCL_T {
                        alpha_t2[State::E_1] -= log_83;
                    } else if self.seq[t + 2] == NUCL_C {
                        alpha_t2[State::E_1] -= log_10;
                    } else if self.seq[t + 2] == NUCL_A {
                        alpha_t2[State::E_1] -= log_07;
                    }

                    let start_freq =
                        self.start_freq_adjustment_start_endminus(t, &self.train.stop1);
                    alpha_t2[State::E_1] -=
                        self.adjusted_probability_from_dist(start_freq, &self.train.e1_dist);
                }
            }

            if num_n > 9 {
                for i in State::iter() {
                    if i != State::R {
                        alphas.t0[i] = f64::MAX;
                        paths.t0[i] = Some(State::R);
                    }
                }
            }
        }

        CalculatedViterbi::new(self.seq, self.train, self.input_cfg, alpha, path)
    }
}

/// Run the specialised Viterbi algorithm.
pub fn viterbi(hmm: &Hmm, train: &Train, seq: &[u8], input_cfg: InputConfig) -> Vec<OutputEntry> {
    let cg_count = {
        // Note: maybe this could be faster with the bytecount crate which has an SSE & AVX2 solution?
        let absolute_count = seq.iter().fold(0, |acc, &character| {
            acc + (character == NUCL_C || character == NUCL_G) as isize
        });

        let idx = ((absolute_count as f64 / seq.len() as f64) * 100.0).floor() as isize - 26;
        clamp(idx, 0, MAX_CG as isize - 1) as usize
    };

    Viterbi::new(hmm, &train.trainings[cg_count], seq, input_cfg)
        .run()
        .backtrack()
}
