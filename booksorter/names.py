"""Name normalization with Serbian (Cyrillic + Latin) awareness."""
import difflib
import re
import unicodedata

_CYR = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E", "Ж": "Ž",
    "З": "Z", "И": "I", "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj", "М": "M", "Н": "N",
    "Њ": "Nj", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "Ћ": "Ć", "У": "U",
    "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č", "Џ": "Dž", "Ш": "Š",
    # Russian extras
    "Й": "J", "Щ": "Šč", "Ы": "Y", "Э": "E", "Ю": "Ju", "Я": "Ja", "Ё": "Jo", "Ъ": "", "Ь": "",
}
_CYR.update({k.lower(): v.lower() for k, v in list(_CYR.items())})
_CYR_TABLE = str.maketrans(_CYR)

_FOLD = str.maketrans({"đ": "dj", "Đ": "Dj"})
_HONORIFICS = re.compile(r"\b(dr|prof|mr|mrs|ms|sir|dame|sv)\.?\s+", re.I)

JUNK_NAMES = {
    "unknown", "nepoznat", "nepoznati autor", "autor", "author", "admin", "administrator",
    "user", "korisnik", "calibre", "ms user", "ms-user", "owner", "windows user", "valued customer",
    "microsoft", "your name", "vlasnik", "various", "razni autori", "grupa autora", "anonymous",
}
JUNK_WORDS = {
    "knjiga", "knjige", "eknjige", "tom", "deo", "dio", "roman", "pripovetke", "izabrana",
    "dela", "djela", "edicija", "biblioteka", "kolekcija", "serijal", "zbirka", "pesme",
    "priče", "price", "epub", "mobi", "pdf", "azw", "format", "formatu",
    # Serbian function words: common in titles, never in names ('Rat i mir', 'Pad u noć')
    "i", "u", "na", "za", "se", "o", "od", "do", "iz", "sa", "s", "k", "ka", "po", "pri", "bez",
    "kroz", "nad", "pod", "je", "su", "ne", "ni", "li", "kao", "sto", "što", "moj", "moja",
    # computer / Word default usernames ('Ms-user', 'HP Owner', 'Windows User')
    "user", "owner", "admin", "administrator", "windows", "microsoft", "korisnik", "vlasnik",
    "pc", "hp", "dell", "lenovo", "acer", "asus", "toshiba", "compaq", "customer",
    # English title words picked up from 'Naslov originala' ('His Parables', 'Poems Prvo')
    "his", "her", "the", "and", "of", "poems", "stories", "tales", "prvo", "izdanje",
    # credit / metadata words glued onto names ('Alexandra Pottrer Translation')
    "translation", "translated", "translator", "edited", "editor", "copyright", "prevod",
    "prevela", "preveo", "prijevod", "urednik", "scan", "skenirao", "obrada", "ebook", "e-book",
}


def transliterate(s: str) -> str:
    # NFC first: macOS filenames store 'ž' as 'z' + combining caron, which breaks every pattern
    return unicodedata.normalize("NFC", s or "").translate(_CYR_TABLE)


def fold(s: str) -> str:
    """Latin, lowercase, no diacritics: 'Нушић' / 'Nušić' / 'Nusic' -> 'nusic'."""
    s = transliterate(s).translate(_FOLD).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z]+", fold(s))


def key(name: str) -> str:
    return " ".join(sorted(tokens(name)))


def display_clean(name: str) -> str:
    name = transliterate(name).strip(" .,-_")
    name = re.sub(r"\b([OoDd])[_’ʼ`´']\s*(?=[A-Z])", r"\1'", name)  # O_Brien / O’ Brien -> O'Brien
    name = _HONORIFICS.sub("", name)
    name = re.sub(r"\b(Dž|DŽ|Dz|DZ|Dj|DJ|Lj|LJ|Nj|NJ|[A-ZČĆŠŽĐ])\.(?=[A-ZČĆŠŽĐ])", r"\1. ", name)  # 'F.M.Dostojevski' -> 'F. M. Dostojevski'

    if name.count(",") == 1:
        last, first = (p.strip() for p in name.split(","))
        name = f"{first} {last}"
    name = re.sub(r"\s+", " ", name.replace("_", " ")).strip()
    if name.isupper() or name.islower():
        name = " ".join(w.capitalize() for w in name.split())
    return name


def has_diacritics(s: str) -> bool:
    return any(c in s for c in "čćšžđČĆŠŽĐ")


def looks_serbian_transcribed(s: str) -> bool:
    """Serbian letters, or their ASCII stand-ins as initials ('Dz. R. R. Martin', 'Dj.')."""
    return has_diacritics(s) or bool(re.search(r"\b(?:D[zžjJ]|Lj|Nj)\.", s))


def looks_like_name(s: str) -> bool:
    if re.search(r"www\.|https?:|\.(com|net|org|rs|hr|ba|me)\b|@", s or "", re.I):
        return False  # web addresses / e-mails from tags ('Www.sftim.com')
    s = display_clean(s)
    if not s or any(ch.isdigit() for ch in s) or fold(s) in JUNK_NAMES:
        return False
    toks = tokens(s)
    if not 2 <= len(toks) <= 4 or len(max(toks, key=len)) < 3:
        return False
    # 'K.', 'S.', 'O' (as in 'Anne O Brien') are initials, not words
    words = [fold(w) for w in s.split() if not (w.endswith(".") or (len(w) == 1 and w.isupper()))]
    return not any(w in JUNK_WORDS for w in words)


_DIGRAPH_INITIALS = {"dz", "lj", "nj", "dj"}  # Serbian one-letter initials: Dž., Lj., Nj., Đ.


def is_initial(tok: str) -> bool:
    return len(tok) == 1 or tok in _DIGRAPH_INITIALS


def initial_sound(tok: str) -> str:
    """First sound, so initials match across scripts: 'Dž.' ~ 'George' ~ 'J.', 'Č.' ~ 'Charles'."""
    if tok[:2] in ("dz", "dj") or (tok[0] == "g" and tok[1:2] in ("e", "i", "y")):
        return "j"
    if tok[:2] == "ch":
        return "c"
    if tok[:2] in ("lj", "nj"):
        return tok[:2]
    return tok[0]


def _initials_match(a: list[str], b: list[str]) -> bool:
    """'j k rowling' ~ 'joanne kathleen rowling', 'dz r r martin' ~ 'george r r martin'."""
    if not a or not b or len(a) != len(b):
        return False
    if a[-1] != b[-1]:  # transcribed surname ('Dikens' ~ 'Dickens'), only against a foreign spelling
        if (_phonetic_token(a[-1]) != _phonetic_token(b[-1])
                or max(foreignness(" ".join(a)), foreignness(" ".join(b))) == 0):
            return False
    return all(x == y or ((is_initial(x) or is_initial(y)) and initial_sound(x) == initial_sound(y))
               for x, y in zip(a[:-1], b[:-1]))


_PHONETIC_RULES = [
    ("ce", "se"), ("ci", "si"), ("cy", "si"), ("dz", "j"), ("dj", "j"),
    ("ge", "je"), ("gi", "ji"),  # soft g: George ~ Džordž
    ("tz", "c"), ("ts", "c"), ("sch", "s"), ("sh", "s"), ("ch", "k"),
    ("ph", "f"), ("th", "t"), ("ck", "k"), ("gh", ""), ("qu", "kv"), ("q", "k"), ("x", "ks"),
    ("w", "v"), ("f", "v"), ("c", "k"),  # f/v: 'Stephen' is transcribed 'Stiven'
]


def _phonetic_token(tok: str) -> str:
    """Consonant skeleton that survives Serbian transcription of foreign names:
    'munro'/'manro' -> 'mnr', 'john'/'dzon' -> 'jn', 'krentz'/'krenc' -> 'krnk'."""
    for a, b in _PHONETIC_RULES:
        tok = tok.replace(a, b)
    head, rest = tok[:1], tok[1:]
    rest = re.sub(r"[aeiouyjhv]", "", rest) if head != "j" else re.sub(r"[aeiouyjh]", "", rest)
    head = "j" if head in "jy" else ("" if head in "aeiou" else head)
    return re.sub(r"(.)\1+", r"\1", head + rest)


def phonetic_key(name: str) -> str:
    return " ".join(sorted(_phonetic_token(t) for t in tokens(name)))


def foreignness(name: str) -> int:
    """How 'original' (non-transcribed) a spelling looks: Alice Munro > Alis Manro."""
    f = name.lower()
    return len(re.findall(r"[wyqx]|[bcdfgkprt]h|hn|c[eiy]|ck|oo|ee|ou|ea|([bcdfgklmnprstz])\1", f))


def _pair_words(ta: list[str], tb: list[str]) -> list[tuple[str, str]]:
    """Pair each word with its closest counterpart ('perez reverte arturo' vs 'arturo perez reverte')."""
    left, pairs = list(tb), []
    for x in ta:
        y = max(left, key=lambda w: difflib.SequenceMatcher(None, x, w).ratio())
        left.remove(y)
        pairs.append((x, y))
    return pairs


def _drop_initials(name: str) -> str:
    return " ".join(t for t in tokens(name) if not is_initial(t))


def _conflicting_initials(a: str, b: str) -> bool:
    """'John A. Smith' vs 'John B. Smith': different initials in the same spot = different people."""
    ta, tb = tokens(a), tokens(b)
    return len(ta) == len(tb) and any(
        is_initial(x) and is_initial(y) and initial_sound(x) != initial_sound(y) for x, y in zip(ta, tb))


def same_person(a: str, b: str) -> bool:
    if _conflicting_initials(a, b):
        return False
    if _same_person(a, b):
        return True
    # middle initials are optional: 'Arthur C. Clarke' ~ 'Artur Klark'
    ta, tb = tokens(a), tokens(b)
    if len(ta) != len(tb):
        # 'Dž. R. R. Martin' ~ 'George Martin': drop only middle initials, keep the first one
        ma = [t for i, t in enumerate(ta) if i in (0, len(ta) - 1) or not is_initial(t)]
        mb = [t for i, t in enumerate(tb) if i in (0, len(tb) - 1) or not is_initial(t)]
        if len(ma) == len(mb) >= 2 and _initials_match(ma, mb):
            return True
        a2, b2 = _drop_initials(a), _drop_initials(b)
        if len(a2.split()) >= 2 and len(b2.split()) >= 2 and (a2, b2) != (" ".join(ta), " ".join(tb)):
            return _same_person(a2, b2, foreign=max(foreignness(a), foreignness(b)))
    return False


def _same_person(a: str, b: str, foreign: int = 0) -> bool:
    ka, kb = key(a), key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    pa, pb = phonetic_key(a), phonetic_key(b)
    # transcription match ('Alis Manro' ~ 'Alice Munro') only if one side looks foreign,
    # so two Serbian names with similar consonants ('Marić'/'Mrak') are never merged
    if (pa == pb and len(pa.replace(" ", "")) >= 4 and len(tokens(a)) == len(tokens(b))
            and max(foreignness(a), foreignness(b), foreign) > 0):
        return True
    # typos ('Sandreson' ~ 'Sanderson'): every word must be close, not just the whole string —
    # 'Stephen King' vs 'Stephen Hawking' is 89% alike as a string but a different person
    ta, tb = sorted(tokens(a)), sorted(tokens(b))
    if len(ta) == len(tb):
        ratios = [difflib.SequenceMatcher(None, x, y).ratio() for x, y in _pair_words(ta, tb)]
        if min(ratios) >= 0.75 and sum(ratios) / len(ratios) >= 0.85:
            return True
    return _initials_match(tokens(a), tokens(b))


_CREDITS = re.compile(
    r"\b(prevod\w*|preve(?:o|la|li)|s \w+ preve\w*|predgovor\w*|pogovor\w*|urednik\w*|urednic\w*|"
    r"uredi(?:o|la)|lektur\w*|korektur\w*|ilustr\w*|dizajn\w*|recenz\w*|priredi\w*|"
    r"izdavac\w*|stampa|tiraz|translated by|edited by|illustrated by|foreword by|introduction by)"
    r"\b[:\s]*(?:\S+\s*){0,5}")


def strip_credits(folded_text: str) -> str:
    """Drop translators, editors, foreword writers… so they don't count as the author."""
    return _CREDITS.sub(" ", folded_text)


def appears_in(name: str, folded_text: str) -> bool:
    """Is this name (any order, fuzzy per token) present in the text? Text must be fold()-ed."""
    toks = [t for t in tokens(name) if len(t) > 1]
    if not toks or not folded_text:
        return False
    words = set(re.findall(r"[a-z]+", folded_text))
    surname_hit = any(_fuzzy_in(t, words) for t in toks[-1:])
    others = sum(_fuzzy_in(t, words) for t in toks[:-1])
    return surname_hit and (others >= 1 or len(toks) == 1)


def _fuzzy_in(tok: str, words: set[str]) -> bool:
    if tok in words:
        return True
    # handle Serbian case endings: 'nusica', 'crnjanskog'
    return any(w.startswith(tok) and len(w) - len(tok) <= 3 for w in words if w[:3] == tok[:3])


def is_title_like(name: str, title: str) -> bool:
    """Is this candidate actually the book title? Only trusts plain titles: a tag title
    like 'Alice Clayton - Moj sused' contains the author too, so it tells us nothing."""
    if not title or re.search(r"\s[-–:]\s|\s[-–]|[-–]\s", title):
        return False
    nt, tt = set(tokens(name)), set(tokens(title))
    return bool(nt and tt) and len(nt & tt) / len(nt | tt) >= 0.6


COMMON_FIRST_NAMES = set("""
ana ivo ivan jovan milan milos marko nikola petar stefan dragan zoran goran dejan vladimir
aleksandar branislav branko danilo dobrica mesa borislav vuk laza momo mirjana svetlana
jelena marija milica dragana vesna ljiljana gordana snezana jasmina isidora desanka biljana
slobodan milorad miodrag radomir dusan dusko mihajlo mihailo vasko ljubomir miroslav ante
slavenka dubravka vida svetislav david danijel filip luka lazar nenad srdjan igor bora
john james robert michael william david richard thomas charles joseph george paul mark peter
stephen dan dean nora ken jack tom alex max leo oscar victor henry edward arthur agatha
mary jane anne ann elizabeth sarah emma alice kate katherine catherine laura lisa julia
nicholas nick patrick sophie sofia helen margaret susan nancy linda karen jennifer jessica
amanda abby abigail adele adriana adrienne agnes amelie amos anders cecelia cecilia candace
carmen cassia dorothy donna virginia ernest franz fjodor fyodor lav lev anton nikolaj
isak daglas greg bil dzon dzejms dzordz dzek dzejn dzesi den ernest tomas majkl stiven
robert ricard vilijam hari meri keti suzan helen ajzak artur aleksandr fjodor lav
lauren lee
stanislav stanislaw kejt kit kejn tomas dzozef dzordz dzejson dzesika dzulija dzudi dzoan
dzeremi dzastin dzefri stiven stefani piter pol mari meri frensis ketrin kerolajn elizabet
majkl mihael rodzer entoni endru hauard harold henri erik edvard ernest franc frederik
bernard vilijam volter lorens lusi lujza helena hana ana nina vera olga irina tatjana natasa
jurij jurica josip ivica mirko slavko zlatko zeljko dragoljub radoje jovan djordje stevan
aleksa laza vojislav rastko momcilo branimir kresimir tomislav drazen marina tanja sanja
mihail maksim gabriel umberto italo paulo jose isabel milan orhan elif danielle nicholas
""".split())


def first_name(name: str) -> str:
    toks = tokens(name)
    return toks[0] if len(toks) >= 2 else ""


def fix_order(name: str, first_names: set[str], order_counts=None) -> str:
    """'Lem Stanislav' -> 'Stanislav Lem'. If exactly one word is a known first name it goes first;
    otherwise the word order seen most often across the library wins."""
    parts = name.split()
    if len(parts) != 2:
        return name
    flipped = f"{parts[1]} {parts[0]}"
    a, b = (fold(p) for p in parts)
    if (a in first_names) != (b in first_names):
        return name if a in first_names else flipped
    if order_counts is not None:
        seen = sum(c for n, c in order_counts.items() if fold(n) == fold(name))
        seen_flipped = sum(c for n, c in order_counts.items() if fold(n) == fold(flipped))
        if seen_flipped > seen:
            return flipped
    return name



_NAME_SEPARATORS = re.compile(r"\s*(?:&|;|/|\+|,)\s*|\s+(?:and|i|und|et|y)\s+|\s+-\s+|(?<=[a-zčćšžđ])-(?=[A-ZČĆŠŽĐ][a-z]+ )")


def split_names(s: str) -> list[str]:
    """'Kathryn Passig&Aleks Scholz' -> ['Kathryn Passig', 'Aleks Scholz']. Only splits when every
    piece is a full name, so 'Rat i mir' or 'Christie, Agatha' stay whole."""
    pieces = [display_clean(p) for p in _NAME_SEPARATORS.split(s or "") if p and p.strip()]
    if len(pieces) >= 2 and all(looks_like_name(p) for p in pieces):
        return pieces
    return [s]
