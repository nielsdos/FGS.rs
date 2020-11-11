mod backtrack;
mod hmm;
mod matrix;
mod nucleotide;
mod output_entry;
mod run;
mod train;
mod util;
mod viterbi;

use crate::hmm::Hmm;
use crate::run::run;
use crate::train::Train;
use rayon::ThreadPoolBuilder;
use std::error::Error;
use std::io;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "compbio-project", about = "FragGeneScan in Rust")]
pub struct Opt {
    /// Set to one for whole genome, zero for short reads.
    #[structopt(short, possible_values = &["0", "1"])]
    whole_genome: u8,

    /// How many worker threads are allowed to be used? Zero will use the amount of logical cores.
    #[structopt(short = "p", default_value = "1")]
    threads: usize,

    /// Path to model parameters, constructed from training.
    #[structopt(short = "t", parse(from_os_str))]
    hmm_path: PathBuf,

    /// Path to where a DNA-FASTA file must be written, if wanted.
    #[structopt(short = "d", parse(from_os_str))]
    dna_out: Option<PathBuf>,

    /// Path to where a metadata file must be written, if wanted.
    #[structopt(short = "e", parse(from_os_str))]
    meta_out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Read & setup options
    let opt: Opt = Opt::from_args();
    ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()?;

    if !opt.hmm_path.is_file() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::Other,
            "Hmm parameter path (-t) is not a file or does not exist",
        )));
    }
    let train_dir = opt.hmm_path.parent().unwrap(); // Can't fail because we already know this is a file.

    // start <- ./train/start (start in forward)
    // stop <- ./train/stop (stop in forward)
    // start1 <- ./train/stop1 (start in reverse)
    // stop1 <- ./train/start1 (stop in reverse)
    let hmm_file = &opt.hmm_path;
    let m_file = train_dir.join("gene");
    let m1_file = train_dir.join("rgene");
    let n_file = train_dir.join("noncoding");
    let start_file = train_dir.join("start");
    let stop_file = train_dir.join("stop");
    let start1_file = train_dir.join("stop1");
    let stop1_file = train_dir.join("start1");
    let d_file = train_dir.join("pwm");

    let hmm = Hmm::from_file(&hmm_file)?;
    let train = Train::from_files(
        &m_file,
        &m1_file,
        &n_file,
        &start_file,
        &stop_file,
        &start1_file,
        &stop1_file,
        &d_file,
    )?;

    run(opt, hmm, train);

    Ok(())
}
