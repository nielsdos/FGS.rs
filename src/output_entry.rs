/// Single output entry for a piece of dna.
#[derive(Debug)]
pub struct OutputEntry {
    dna: Option<Box<[u8]>>,
    protein: Box<[u8]>,
    start: usize,
    end: usize,
    score: f64,
    frame: usize,
    positive_strand: bool,
    insertions: Box<[usize]>,
    deletions: Box<[usize]>,
}

impl OutputEntry {
    /// Constructs a new output entry.
    pub fn new(
        dna: Option<Vec<u8>>,
        protein: Vec<u8>,
        start: usize,
        end: usize,
        score: f64,
        frame: usize,
        positive_strand: bool,
        insertions: Vec<usize>,
        deletions: Vec<usize>,
    ) -> Self {
        Self {
            dna: dna.map(|dna| dna.into_boxed_slice()),
            protein: protein.into_boxed_slice(),
            start,
            end,
            score,
            frame,
            positive_strand,
            insertions: insertions.into_boxed_slice(),
            deletions: deletions.into_boxed_slice(),
        }
    }

    /// Gets the DNA.
    #[inline]
    pub fn dna(&self) -> Option<&[u8]> {
        self.dna.as_ref().map(|dna| &dna[..])
    }

    /// Gets the protein.
    #[inline]
    pub fn protein(&self) -> &[u8] {
        &self.protein
    }

    /// Gets a description for the FASTA entry.
    pub fn desc(&self) -> String {
        format!(
            "{}_{}_{}",
            self.start,
            self.end,
            if self.positive_strand { '+' } else { '-' }
        )
    }

    /// Gets the metadata.
    pub fn meta(&self) -> String {
        format!(
            "{}\t{}\t{}\t{:.6}\tI:{}\tD:{}\n",
            self.start,
            self.end,
            if self.positive_strand { '+' } else { '-' },
            self.score,
            self.insertions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
            self.deletions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )
    }
}
