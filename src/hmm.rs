use crate::nucleotide::{NUCL_A, NUCL_C, NUCL_G, NUCL_T, NUCL_UNKNOWN};
use crate::viterbi::STATE_COUNT;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::iter::Iterator;
use std::ops;
use std::path::Path;
use std::str::SplitWhitespace;
use strum_macros::EnumCount;

pub type InitProbDist = [f64; STATE_COUNT];
pub type TransitionProb = [f64; TRANSITION_COUNT];
pub type TransBase = [[f64; 4]; 4];

/// Transitions in the Hmm.
#[repr(usize)]
#[derive(EnumCount)]
pub enum Transition {
    MM = 0,
    MI = 1,
    MD = 2,
    II = 3,
    IM = 4,
    DD = 5,
    DM = 6,
    GE = 7,
    GG = 8,
    ER = 9,
    RS = 10,
    RR = 11,
    ES = 12,
    ES1 = 13,
}

impl<T> ops::Index<Transition> for [T] {
    type Output = T;

    fn index(&self, index: Transition) -> &Self::Output {
        &self[index as usize]
    }
}

impl<T> ops::IndexMut<Transition> for [T] {
    fn index_mut(&mut self, index: Transition) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

/// Hidden Markov Model transitions and initial probability distribution.
pub struct Hmm {
    /// The initial state distribution.
    pub pi: InitProbDist,
    /// Transition probability from one state to another.
    pub tr: TransitionProb,
    /// Transition probability of insert -> insert for a nucleobase.
    pub tr_i_i: TransBase,
    /// Transition probability of match -> insert for a nucleobase.
    pub tr_m_i: TransBase,
}

/// Convert transition in HMM to index.
fn tr_to_num(s: &str) -> Transition {
    match s {
        "MM" => Transition::MM,
        "MI" => Transition::MI,
        "MD" => Transition::MD,
        "II" => Transition::II,
        "IM" => Transition::IM,
        "DD" => Transition::DD,
        "DM" => Transition::DM,
        "GE" => Transition::GE,
        "GG" => Transition::GG,
        "ER" => Transition::ER,
        "RS" => Transition::RS,
        "RR" => Transition::RR,
        "ES" => Transition::ES,
        "ES1" => Transition::ES1,
        _ => panic!("unknown state transition {}", s),
    }
}

/// Convert nucleobase to number/index.
fn base_to_num(c: &str) -> usize {
    (match c {
        "A" | "a" => NUCL_A,
        "C" | "c" => NUCL_C,
        "G" | "g" => NUCL_G,
        "T" | "t" => NUCL_T,
        _ => NUCL_UNKNOWN,
    }) as usize
}

impl Hmm {
    // Helper to read some parts
    fn process<T>(
        lines: &mut Lines<BufReader<File>>,
        n: usize,
        dest: &mut T,
        processor: &dyn Fn(usize, SplitWhitespace, f64, &mut T) -> (),
    ) -> Result<(), Box<dyn Error>> {
        for (i, line) in lines.skip(1).take(n).enumerate() {
            let line = line?;
            let mut split = line.split_whitespace();
            let prob = split.next_back().expect("split").parse::<f64>()?.ln();
            processor(i, split, prob, dest);
        }
        Ok(())
    }

    /// Construct Hmm from a file.
    pub fn from_file(hmm_file: &Path) -> Result<Hmm, Box<dyn Error>> {
        let hmm_file = File::open(hmm_file)?;
        let reader = BufReader::new(hmm_file);
        let mut lines = reader.lines();

        let mut tr = TransitionProb::default();
        let mut tr_m_i = TransBase::default();
        let mut tr_i_i = TransBase::default();
        let mut pi = InitProbDist::default();
        Self::process(&mut lines, 14, &mut tr, &|_, mut s, p, d| {
            d[tr_to_num(s.next().expect("tr"))] = p;
        })?;
        let two_part = |_, mut s: SplitWhitespace, p: f64, d: &mut TransBase| {
            let start = s.next();
            let end = s.next();
            d[base_to_num(start.expect("start"))][base_to_num(end.expect("end"))] = p;
        };
        Self::process(&mut lines, 16, &mut tr_m_i, &two_part)?;
        Self::process(&mut lines, 16, &mut tr_i_i, &two_part)?;
        Self::process(&mut lines, STATE_COUNT, &mut pi, &|i, _, p, d| {
            d[i] = -p;
        })?;

        Ok(Hmm {
            pi,
            tr,
            tr_i_i,
            tr_m_i,
        })
    }
}
