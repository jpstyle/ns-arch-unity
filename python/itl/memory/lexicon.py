class Lexicon:
    """
    Lexicon of open-classed words <-> their denoted concepts in physical world. Should
    allow many-to-many mappings between symbols and denotations.
    """
    # Special reserved symbols; any predicates obtainable from the parsing-translation
    # pipeline, that do NOT refer to first-order domain concepts
    RESERVED = [
        # Invokes object identity check
        ("=", "*"),
        # Invokes concept instance (i.e., set element) check
        ("isinstance", "*"),
        # Invokes supertype-subtype check (against taxonomy KB)
        ("subtype", "*"),
        # Invokes concept difference computation
        ("diff", "*"),
        # Pronoun indicator
        ("pronoun", "*"),
        # Expresses a dialogue participant's belief on a statement w/ irrealis mood
        ("think", "*")
    ]

    def __init__(self):
        self.s2d = {}     # Symbol-to-denotation
        self.d2s = {}     # Denotation-to-symbol
        self.d_freq = {}  # Denotation frequency

        # Add reserved symbols & denotations
        for r in Lexicon.RESERVED: self.add(r, r)

        # (Temp) Inventory of relation concept is a fixed singleton set, containing "have"
        self.add(("have", "v"), (0, "rel"))

    def __repr__(self):
        return f"Lexicon(len={len(self.s2d)})"

    def __contains__(self, symbol):
        return symbol in self.s2d

    def add(self, symbol, denotation, freq=None):
        # For consistency; we don't need the 'adjective satellite' thingy
        # from WordNet
        if symbol[1] == "s": symbol = (symbol[0], "a")

        # Symbol-to-denotation
        if symbol in self.s2d:
            self.s2d[symbol].append(denotation)
        else:
            self.s2d[symbol] = [denotation]
        
        # Denotation-to-symbol
        if denotation in self.d2s:
            self.d2s[denotation].append(symbol)
        else:
            self.d2s[denotation] = [symbol]

        freq = freq or 1
        self.d_freq[denotation] = freq
