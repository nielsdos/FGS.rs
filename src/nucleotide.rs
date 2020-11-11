pub const NUCL_A: u8 = 0;
pub const NUCL_C: u8 = 1;
pub const NUCL_G: u8 = 2;
pub const NUCL_T: u8 = 3;
pub const NUCL_UNKNOWN: u8 = 4;

/// Trinucleotide index. None in case of invalid.
fn trinucleotide_raw(a: u8, b: u8, c: u8) -> Option<u8> {
    if a == NUCL_UNKNOWN || b == NUCL_UNKNOWN || c == NUCL_UNKNOWN {
        None
    } else {
        Some((a << 4) | (b << 2) | c)
    }
}

/// Trinucleotide index
pub fn trinucleotide(a: u8, b: u8, c: u8) -> usize {
    trinucleotide_raw(a, b, c).unwrap_or(0) as usize
}

/// Trinucleotide index for PEP
pub fn trinucleotide_pep(a: u8, b: u8, c: u8) -> usize {
    trinucleotide_raw(a, b, c).unwrap_or(64) as usize
}

/// Protein from DNA strand.
pub fn protein(dna: &[u8], positive_strand: bool, whole_genome: bool) -> Vec<u8> {
    let codon_code = b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLFX";
    let anti_codon_code = b"FVLICGRSSAPTYDHNLVLMWGRRSAPT*EQKFVLICGRSSAPTYDHNLVLI*GRRSAPT*EQKX";

    // Select correct lookup based on if it's a positive or negative strand.
    let lut = if positive_strand {
        codon_code
    } else {
        anti_codon_code
    };

    let iterator = dna
        .chunks_exact(3)
        .map(|chunk| lut[trinucleotide_pep(chunk[0], chunk[1], chunk[2])]);

    let mut vec = if positive_strand {
        iterator.collect::<Vec<_>>()
    } else {
        iterator.rev().collect::<Vec<_>>()
    };

    // Remove ending *
    if let Some(&b'*') = vec.last() {
        vec.pop();
    }

    // Alternative start codons still encode for Met.
    // E. coli uses 83% AUG (3542/4284), 14% (612) GUG, 3% (103) UUG and one or two others (e.g., an AUU and possibly a CUG).
    // Only consider two major alternative ones, GTG and TTG.
    if whole_genome {
        if positive_strand {
            let s = trinucleotide_pep(dna[0], dna[1], dna[2]);
            if s == trinucleotide_pep(NUCL_G, NUCL_T, NUCL_G)
                || s == trinucleotide_pep(NUCL_T, NUCL_T, NUCL_G)
            {
                vec[0] = b'M';
            }
        } else {
            let len = dna.len();
            let s = trinucleotide_pep(dna[len - 3], dna[len - 2], dna[len - 1]);
            if s == trinucleotide_pep(NUCL_C, NUCL_A, NUCL_C)
                || s == trinucleotide_pep(NUCL_C, NUCL_A, NUCL_A)
            {
                vec[0] = b'M';
            }
        }
    }

    vec
}

/// Check if the symbol is in the ACGT alphabet.
pub fn is_acgt(x: u8) -> bool {
    x == b'A' || x == b'C' || x == b'G' || x == b'T'
}

/// Convert nucleobase to number/index.
pub fn base_to_num_unchecked(symbol: u8) -> u8 {
    debug_assert!(is_acgt(symbol));
    // Using a match arm actually reduces performance.
    // If you multiply the ASCII value with 3, you get following values:
    // A 110 00 011
    // C 110 01 001
    // G 110 10 101
    // T 111 11 100
    //       ^^
    let symbol = symbol * 3;
    (symbol >> 3) & 3
}

/// Convert the DNA numbers/indices back to letters.
pub fn convert_dna_nrs_to_letters(dna: &mut Vec<u8>) {
    for symbol in dna.iter_mut() {
        *symbol = [b'A', b'C', b'G', b'T', b'N'][*symbol as usize];
    }
}
