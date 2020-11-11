use std::default::Default;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const MAX_CG: usize = 44;

pub type TrainTransSingle = [[[f64; 4]; 16]; 6];
pub type TrainStartStopSingle = [[f64; 64]; 61];
pub type TrainNoncodingSingle = [[f64; 4]; 4];
pub type TrainDistSingle = [f64; 6];

/// Training data voor een enkele CG verhouding.
pub struct TrainSingle {
    pub trans: TrainTransSingle,
    pub rtrans: TrainTransSingle,
    pub noncoding: TrainNoncodingSingle,
    pub start: Box<TrainStartStopSingle>,
    pub stop: Box<TrainStartStopSingle>,
    pub start1: Box<TrainStartStopSingle>,
    pub stop1: Box<TrainStartStopSingle>,
    pub s_dist: TrainDistSingle,
    pub e_dist: TrainDistSingle,
    pub s1_dist: TrainDistSingle,
    pub e1_dist: TrainDistSingle,
}

impl Default for TrainSingle {
    fn default() -> Self {
        Self {
            trans: [[[0.0; 4]; 16]; 6],
            rtrans: [[[0.0; 4]; 16]; 6],
            noncoding: [[0.0; 4]; 4],
            start: Box::new([[0.0; 64]; 61]),
            stop: Box::new([[0.0; 64]; 61]),
            start1: Box::new([[0.0; 64]; 61]),
            stop1: Box::new([[0.0; 64]; 61]),
            s_dist: TrainDistSingle::default(),
            e_dist: TrainDistSingle::default(),
            s1_dist: TrainDistSingle::default(),
            e1_dist: TrainDistSingle::default(),
        }
    }
}

/// Training data.
pub struct Train {
    pub trainings: [TrainSingle; MAX_CG],
}

impl Train {
    /// Log-probabiliteit
    fn prob_ln(input: f64) -> f64 {
        input.ln()
    }

    /// Probabiliteit staat al goed
    fn prob_id(input: f64) -> f64 {
        input
    }

    /// Helper for reading the file.
    fn read_helper<P>(
        file: &Path,
        p_max: usize,
        j_max: usize,
        mut processor: P,
        prob_translator: fn(f64) -> f64,
    ) -> Result<(), Box<dyn Error>>
    where
        P: FnMut(usize, usize, usize, f64) -> (),
    {
        let file = File::open(file).expect("file");
        let reader = BufReader::new(file);
        let lines = &mut reader.lines();

        for p in 0..p_max {
            lines.next().expect("line")?;

            for (j, line) in lines.take(j_max).enumerate() {
                let line = line?;
                for (k, num) in line.split_whitespace().enumerate() {
                    processor(
                        p,
                        j,
                        k,
                        prob_translator(num.parse::<f64>().expect("number")),
                    );
                }
            }
        }

        Ok(())
    }

    /// Read training data from files.
    pub fn from_files(
        m_file: &Path,
        m1_file: &Path,
        n_file: &Path,
        start_file: &Path,
        stop_file: &Path,
        start1_file: &Path,
        stop1_file: &Path,
        d_file: &Path,
    ) -> Result<Train, Box<dyn Error>> {
        let mut train = Train {
            // Very ugly, but easiest solution only using safe Rust without external crates.
            // Didn't want to use a Vec to keep memory close together.
            trainings: [
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
                TrainSingle::default(),
            ],
        };

        Self::read_helper(
            m_file,
            MAX_CG,
            6 * 16,
            |p, j, k, prob| train.trainings[p].trans[j / 16][j % 16][k] = prob,
            Self::prob_ln,
        )?;
        Self::read_helper(
            m1_file,
            MAX_CG,
            6 * 16,
            |p, j, k, prob| train.trainings[p].rtrans[j / 16][j % 16][k] = prob,
            Self::prob_ln,
        )?;
        Self::read_helper(
            start_file,
            MAX_CG,
            61,
            |p, j, k, prob| train.trainings[p].start[j][k] = prob,
            Self::prob_ln,
        )?;
        Self::read_helper(
            stop_file,
            MAX_CG,
            61,
            |p, j, k, prob| train.trainings[p].stop[j][k] = prob,
            Self::prob_ln,
        )?;
        Self::read_helper(
            start1_file,
            MAX_CG,
            61,
            |p, j, k, prob| train.trainings[p].start1[j][k] = prob,
            Self::prob_ln,
        )?;
        Self::read_helper(
            stop1_file,
            MAX_CG,
            61,
            |p, j, k, prob| train.trainings[p].stop1[j][k] = prob,
            Self::prob_ln,
        )?;
        Self::read_helper(
            n_file,
            MAX_CG,
            4,
            |p, j, k, prob| train.trainings[p].noncoding[j][k] = prob,
            Self::prob_ln,
        )?;
        Self::read_helper(
            d_file,
            MAX_CG,
            4,
            |p, j, k, prob| {
                let lut = [
                    &mut train.trainings[p].s_dist,
                    &mut train.trainings[p].e_dist,
                    &mut train.trainings[p].s1_dist,
                    &mut train.trainings[p].e1_dist,
                ];
                lut[j][k] = prob
            },
            Self::prob_id,
        )?;

        Ok(train)
    }
}
