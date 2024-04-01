def generate_dna_kmers(k):
    '''
    Return a list of all possible substrings of
    length k using only characters A, C, T, and G
    '''
    bases = ["A", "C", "T", "G"]

    last = bases
    current = []
    for i in range(k-1):
        for b in bases:
            for l in last:
                current.append(l+b)
        last = current
        current= []
    return last

def count_mer(mer, seq):
    '''
    Counts the number of times a substring mer
    ocurrs in the sequence seq (including overlapping
    occurrences)

    sample use: count_mer("GGG", "AGGGCGGG") => 2
    '''

    k = len(mer)
    count = 0
    for i in range(0, len(seq)-k+1):
        if mer == seq[i:i+k]:
            count = count + 1
    return count

def kmer_count(k, seq):
    '''
    Return a list of the number of times each possible k-mer appears
    in seq, including overlapping occurrences.
    '''

    mers = generate_dna_kmers(k)
    rv = []
    for m in mers:
        cnt = count_mer(m, seq)
        if cnt != 0:
            rv.append((m, cnt))
    return rv


with open("String_data.txt", "r") as f:
    inter = f.read()
    dat = ''.join(inter.splitlines())

print(kmer_count(3, dat))