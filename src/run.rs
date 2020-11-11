use crate::hmm::Hmm;
use crate::nucleotide::{base_to_num_unchecked, is_acgt, NUCL_UNKNOWN};
use crate::output_entry::OutputEntry;
use crate::train::Train;
use crate::util::upper;
use crate::{viterbi, Opt};
use bio::io::fasta;
use rayon::prelude::{ParallelBridge, ParallelIterator};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::mpsc::channel;
use std::{io, thread};

/// Minimum accepted fragment length.
/// We can't really do meaningful predictions if we have 70 bp or less.
pub const MIN_ACCEPTED_LEN: usize = 71;

/// The struct that holds the data specialised for the provided input.
/// This is just a "plain old data" struct, which is why the fields are public.
#[derive(Debug, Copy, Clone)]
pub struct InputConfig {
    /// Is this input a whole genome (false means short read)?
    pub whole_genome: bool,
    /// Should we save metadata?
    pub save_meta: bool,
    /// Should we save the DNA?
    pub save_dna: bool,
}

/// A single entry to write to the output stream.
/// This single entry links an input name to zero or more outputs.
/// Outputs being DNA, amino acid, metadata.
#[derive(Debug)]
struct WriteEntry {
    /// The list of outputs.
    outputs: Vec<OutputEntry>,
    /// The name of the corresponding sequence.
    name: String,
}

/// This is a tuple to guarantee ordering for `WriteEntry`.
struct WriteTuple {
    /// The index: gives the ordering.
    idx: usize,
    /// The `WriteEntry`.
    data: WriteEntry,
}

impl Eq for WriteTuple {}

impl PartialEq for WriteTuple {
    fn eq(&self, other: &Self) -> bool {
        other.idx.eq(&self.idx)
    }
}

impl PartialOrd for WriteTuple {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // We want to have a min-heap.
        other.idx.partial_cmp(&self.idx)
    }
}

impl Ord for WriteTuple {
    fn cmp(&self, other: &Self) -> Ordering {
        // We want to have a min-heap.
        other.idx.cmp(&self.idx)
    }
}

/// Actually runs the main loop of the program.
pub fn run(opt: Opt, hmm: Hmm, train: Train) {
    let input_cfg = InputConfig {
        whole_genome: opt.whole_genome == 1,
        save_meta: opt.meta_out.is_some(),
        save_dna: opt.dna_out.is_some(),
    };

    let (tx, rx) = channel::<WriteTuple>();

    // We might be working with a lot of data.
    // So instead of collecting it sequentially at the end, we send it to a separate writing thread.
    let writer_thread = thread::spawn(move || {
        let mut faa_out = fasta::Writer::new(io::stdout());

        let mut dna_out = opt
            .dna_out
            .map(|filename| fasta::Writer::to_file(filename).expect("dna output file"));

        // Not a FASTA file, so we use a buffered writer.
        let mut meta_out = opt
            .meta_out
            .map(|filename| BufWriter::new(File::create(filename).expect("meta output file")));

        // We use a binary heap to get the incoming data in the right order.
        // While data may wait here for a bit until it is written, it shouldn't be too bad.
        let mut heap = BinaryHeap::<WriteTuple>::new();
        let mut next_wanted = 0;

        loop {
            // Work ordered, only continue if the top is the expected next value.
            while {
                if let Some(e) = heap.peek() {
                    e.idx == next_wanted
                } else {
                    false
                }
            } {
                let entry = heap.pop().unwrap().data;
                for output in entry.outputs {
                    let desc = output.desc();

                    // Amino Acid output
                    faa_out
                        .write(&entry.name, Some(&desc), output.protein())
                        .expect("write faa");

                    // DNA output
                    if let Some(dna_out) = dna_out.as_mut() {
                        dna_out
                            .write(
                                &entry.name,
                                Some(&desc),
                                output.dna().expect("dna should exist"),
                            )
                            .expect("write dna");
                    }

                    // Metadata output
                    if let Some(meta_out) = meta_out.as_mut() {
                        meta_out
                            .write_all(format!(">{}\n", entry.name).as_bytes())
                            .expect("write meta");
                        meta_out
                            .write_all(output.meta().as_bytes())
                            .expect("write meta");
                    }
                }

                next_wanted += 1;
            }

            // Can only fail if all senders are broken and no messages are left in the buffer.
            let msg = match rx.recv() {
                Ok(msg) => msg,
                Err(_) => return,
            };

            heap.push(msg);
        }
    });

    // Process records.
    fasta::Reader::new(io::stdin())
        .records()
        .map(|record| record.expect("record should exist"))
        .filter(|record| record.seq().len() >= MIN_ACCEPTED_LEN)
        .enumerate()
        .par_bridge()
        .for_each_with(tx, |tx, (idx, record)| {
            // Can't modify in-place sadly.
            let seq = record
                .seq()
                .iter()
                .map(|&symbol| {
                    let symbol = upper(symbol);
                    if is_acgt(symbol) {
                        base_to_num_unchecked(symbol)
                    } else {
                        NUCL_UNKNOWN
                    }
                })
                .collect::<Vec<_>>();

            let data = WriteEntry {
                outputs: viterbi::viterbi(&hmm, &train, &seq, input_cfg),
                name: String::from(record.id()),
            };

            // Sending can only fail if the receiving end is broken.
            tx.send(WriteTuple { idx, data }).expect("send");
        });

    writer_thread.join().unwrap();
}
