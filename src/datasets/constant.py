LABEL_OTHERS = "OTHERS"

DATA_PATH = "./data"
UD_DATA_PATH = DATA_PATH + "/ud-treebanks-v2.8"

UD_POS_LABELS = [
    "_", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
    "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
    "PUNCT", "SCONJ", "SYM", "VERB", "X"
]

UD_HEAD_LABELS = [
    "_", "acl", "advcl", "advmod", "amod", "appos",
    "aux", "case", "cc", "ccomp", "clf", "compound",
    "conj", "cop", "csubj", "dep", "det", "discourse",
    "dislocated", "expl", "fixed", "flat", "goeswith",
    "iobj", "list", "mark", "nmod", "nsubj", "nummod",
    "obj", "obl", "orphan", "parataxis", "punct",
    "reparandum", "root", "vocative", "xcomp"
]

ISO638_TO_LANG = {
    "ar": "Arabic",         # Semitic, Afro-Asiatic
    "bg": "Bulgarian",      # Slavic, Indo-European
    "cs": "Czech",          # Slavic, Indo-European
    "ca": "Catalan",        # Romance, Indo-European
    "de": "German",         # Germanic, Indo-European
    "el": "Greek",          # Greek, Indo-European
    "en": "English",        # Germanic, Indo-European
    "es": "Spanish",        # Romance, Indo-European
    "et": "Estonian",       # Finnic, Uralic
    "fa": "Persian",        # Iranian, Indo-European
    "fi": "Finnish",        # Finnic, Uralic
    "fr": "French",         # Romance, Indo-European
    "he": "Hebrew",         # Semitic, Afro-Asiatic
    "hi": "Hindi",          # Indic, Indo-European
    "hu": "Hungarian",      # Ugric, Uralic
    "it": "Italian",        # Romance, Indo-European
    "ja": "Japanese",       # Japanese, Japanese
    "ko": "Korean",         # Korean, Korean
    "lv": "Latvian",        # Baltic, Indo-European
    "nl": "Dutch",          # Germanic, Indo-European
    "no": "Norwegian",
    "pl": "Polish",         # Slavic, Indo-European
    "pt": "Portuguese",     # Romance, Indo-European
    "ro": "Romanian",       # Romance, Indo-European
    "ru": "Russian",        # Slavic, Indo-European
    "sk": "Slovak",
    "ta": "Tamil",          # Southern Dravidian, Dravidian
    "tr": "Turkish",        # Turkic, Altaic
    "ur": "Urdu",           # Indic, Indo-European
    "vi": "Vietnamese",     # Viet-Muong, Austro-Asiatic
    "zh": "Chinese",
    "zh_simp": "Chinese-Simp"
}

UD_ISO638_TO_LANG = {
    "ar": "Arabic-PADT",        # Semitic, Afro-Asiatic
    "bg": "Bulgarian-BTB",      # Slavic, Indo-European
    "cs": "Czech-PDT",          # Slavic, Indo-European
    "ca": "Catalan-AnCora",     # Romance, Indo-European
    "de": "German-GSD",         # Germanic, Indo-European
    "el": "Greek-GDT",          # Greek, Indo-European
    "en": "English-EWT",        # Germanic, Indo-European
    "es": "Spanish-GSD",        # Romance, Indo-European
    "et": "Estonian-EDT",       # Finnic, Uralic
    "fa": "Persian-PerDT",      # Iranian, Indo-European
    "fi": "Finnish-TDT",        # Finnic, Uralic
    "fr": "French-GSD",         # Romance, Indo-European
    "he": "Hebrew-HTB",         # Semitic, Afro-Asiatic
    "hi": "Hindi-HDTB",         # Indic, Indo-European
    "hu": "Hungarian-Szeged",   # Ugric, Uralic
    "it": "Italian-VIT",        # Romance, Indo-European
    "ja": "Japanese-GSD",       # Japanese, Japanese
    "ko": "Korean-Kaist",       # Korean, Korean
    "lv": "Latvian-LVTB",       # Baltic, Indo-European
    "nl": "Dutch-Alpino",       # Germanic, Indo-European
    "no": "Norwegian-Nynorsk",
    "pl": "Polish-PDB",         # Slavic, Indo-European
    "pt": "Portuguese-GSD",     # Romance, Indo-European
    "ro": "Romanian-RRT",       # Romance, Indo-European
    "ru": "Russian-GSD",        # Slavic, Indo-European
    "sk": "Slovak-SNK",
    "ta": "Tamil-TTB",          # Southern Dravidian, Dravidian
    "tr": "Turkish-BOUN",       # Turkic, Altaic
    "ur": "Urdu-UDTB",          # Indic, Indo-European
    "vi": "Vietnamese-VTB",     # Viet-Muong, Austro-Asiatic
    "zh": "Chinese-GSD",
    "zh_simp": "Chinese-GSDSimp"
}
