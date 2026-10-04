import csv
import json
import math
import numpy as np

CSV_FILES = [
    "/Users/4c/Desktop/GHQ/data/loto7_4696_k79.csv",
    # "/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_2970.csv",
    # "/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_plus_1726.csv",
]

TOTAL = math.comb(39, 7)

# Jačina uticaja sličnosti prethodna dva izvlačenja.
GAMMAS = np.array([0.0, 0.5, 1.0, 2.0])

# Jačina koncentracije oko istorijskih narednih hipergrana.
BETAS = np.array([0.15, 0.35, 0.7, 1.2])

SHELL_SIZES = np.array([
    math.comb(7, r) * math.comb(32, 7 - r)
    for r in range(8)
])

BEAM = 8
SEARCH_STEPS = 4


def load(path):
    rows = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        for number, row in enumerate(csv.reader(f), 1):
            if not row:
                continue

            values = sorted(map(int, row))

            if (
                len(values) != 7
                or len(set(values)) != 7
                or min(values) < 1
                or max(values) > 39
            ):
                raise ValueError(
                    f"{path}, red {number}: neispravna kombinacija"
                )

            rows.append(values)

    if len(rows) < 100:
        raise ValueError("Potrebno je najmanje 100 izvlačenja.")

    # Svaki bit predstavlja konkretan čvor 1–39.
    # Jedna maska predstavlja celu hipergranu sa sedam čvorova.
    masks = np.array([
        sum(1 << (x - 1) for x in row)
        for row in rows
    ], dtype=np.uint64)

    return rows, masks


def overlap_matrix(masks):
    result = np.empty(
        (len(masks), len(masks)),
        dtype=np.uint8,
    )

    for start in range(0, len(masks), 256):
        result[start:start + 256] = np.bitwise_count(
            masks[start:start + 256, None] & masks[None, :]
        )

    return result


def evaluate(overlaps, start, stop, gammas, betas):
    """
    Istorijski primer j:
        (E[j-2], E[j-1]) -> E[j]

    Težina primera zavisi od preklapanja njegovog konteksta
    sa konkretnim hipergranama pre ciljnog izvlačenja t.

    Svaka prognoza koristi samo primere j < t.
    """
    normalizers = (
        np.exp(betas[:, None] * np.arange(8)) @ SHELL_SIZES
    )

    result = np.empty(
        (stop - start, len(betas), len(gammas))
    )

    for index, t in enumerate(range(start, stop)):
        context = (
            overlaps[t - 1, 1:t - 1].astype(float)
            + 0.5 * overlaps[t - 2, :t - 2]
        )

        weights = np.exp(
            context[:, None] * gammas[None, :]
        )

        kernels = np.exp(
            betas[:, None] * overlaps[t, 2:t][None, :]
        )

        probability = (
            (kernels @ weights)
            / weights.sum(axis=0)[None, :]
        )
        probability /= normalizers[:, None]

        # Pozitivno: bolji log-score od ravnomerne raspodele.
        result[index] = np.log(probability * TOTAL)

    return result


def decode(mask):
    return [
        x + 1
        for x in range(39)
        if int(mask) & (1 << x)
    ]


def next_candidate(masks, gamma, beta):
    context = (
        np.bitwise_count(
            masks[-1] & masks[1:-1]
        ).astype(float)
        + 0.5 * np.bitwise_count(
            masks[-2] & masks[:-2]
        )
    )

    weights = np.exp(gamma * context)
    weights /= weights.sum()

    # Konkretne istorijske naredne hipergrane.
    targets = masks[2:]
    kernel = np.exp(beta * np.arange(8))
    normalizer = float(kernel @ SHELL_SIZES)

    cache = {}

    def score(candidates):
        missing = sorted(
            set(map(int, candidates)) - cache.keys()
        )

        for start in range(0, len(missing), 256):
            block = missing[start:start + 256]
            values = np.array(block, dtype=np.uint64)

            intersections = np.bitwise_count(
                values[:, None] & targets[None, :]
            )

            probabilities = (
                kernel[intersections] @ weights
            ) / normalizer

            cache.update(
                zip(block, map(float, probabilities))
            )

    def leaders():
        # Jednake ocene rešavaju se deterministički.
        return sorted(
            cache,
            key=lambda mask: (-cache[mask], mask),
        )[:BEAM]

    # Početni kandidati: sve različite istorijske hipergrane.
    score(masks)

    # Zatim istražujemo nove hipergrane zamenom jednog čvora.
    # Ovo nije iscrpna pretraga svih mogućih kombinacija.
    for _ in range(SEARCH_STEPS):
        old = leaders()
        neighbors = set()

        for mask in old:
            inside = [
                i for i in range(39)
                if mask & (1 << i)
            ]
            outside = [
                i for i in range(39)
                if not mask & (1 << i)
            ]

            for removed in inside:
                for added in outside:
                    new_mask = (
                        mask
                        ^ (1 << removed)
                        ^ (1 << added)
                    )
                    neighbors.add(new_mask)

        score(neighbors)

        if leaders() == old:
            break

    best = leaders()[0]
    candidate = decode(best)

    assert len(candidate) == 7

    return candidate, cache[best], len(cache)


def run(path):
    rows, masks = load(path)
    overlaps = overlap_matrix(masks)

    # CSV je hronološki: poslednji red je najnovije izvlačenje.
    # Prvih 60% daje početnu istoriju.
    # Sledećih 20% bira parametre, poslednjih 20% testira model.
    first = int(len(rows) * 0.6)
    second = int(len(rows) * 0.8)

    validation = evaluate(
        overlaps, first, second, GAMMAS, BETAS
    )

    b, g = np.unravel_index(
        np.argmax(validation.mean(axis=0)),
        (len(BETAS), len(GAMMAS)),
    )

    gamma = float(GAMMAS[g])
    beta = float(BETAS[b])

    # Parametri su fiksirani pre testa.
    # Tokom testa prethodno ostvarena izvlačenja postaju istorija.
    test = evaluate(
        overlaps,
        second,
        len(rows),
        np.array([gamma]),
        np.array([beta]),
    )[:, 0, 0]

    mean = float(test.mean())

    # Približan opisni interval; ne koriguje vremensku zavisnost.
    margin = float(
        1.96 * test.std(ddof=1) / math.sqrt(len(test))
    )

    # Za NEXT koristimo svu sada poznatu istoriju ovog CSV-a.
    candidate, probability, examined = next_candidate(
        masks, gamma, beta
    )

    return {
        "CSV": path,
        "broj_izvlacenja": len(rows),
        "poslednja": rows[-1],
        "NEXT": candidate,
        "gamma": gamma,
        "beta": beta,
        "test_broj": len(test),
        "test_prosecni_log_dobitak": mean,
        "test_priblizni_95_interval": [
            mean - margin,
            mean + margin,
        ],
        "model_verovatnoca_NEXT": probability,
        "odnos_prema_uniformnoj": probability * TOTAL,
        "pojedinacno_ocenjenih_kombinacija": examined,
    }


if __name__ == "__main__":
    for csv_path in CSV_FILES:
        print(
            json.dumps(
                run(csv_path),
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


    


"""
pojedinacno_ocenjenih_kombinacija prikazuje stvarni broj različitih kandidata koje je pretraga ocenila

model razlikuje konkretne hipergrane i ocenjuje ih prema distribuciji i vremenskim odnosima u svakom CSV-u

Model čuva konkretne brojeve u hipergranama, 
uči iz istorijskih prelaza i pojedinačno ocenjuje kandidatske kombinacije. 
Nema nasumičnog izbora pomoću seed.

CSV	NEXT
4.696 izvlačenja	6, 10, 12, 13, 23, 32, 39
2.970 izvlačenja	2, 10, 22, 29, 30, 33, 39
1.726 izvlačenja	2, 7, 8, 11, 23, 26, 34


Pretraga je približna; prediktivna prednost još nije potvrđena testovima. 
Sve tri CSV putanje su ugrađene. 
Potreban je numpy>=2.0

numpy                        1.26.4

Successfully installed numpy-2.4.6
"""



"""
{
  "CSV": "/Users/4c/Desktop/GHQ/data/loto7_4696_k79.csv",
  "broj_izvlacenja": 4696,
  "poslednja": [
    5,
    6,
    16,
    21,
    23,
    26,
    36
  ],
  "NEXT": [
    6,
    10,
    12,
    13,
    23,
    32,
    39
  ],
  "gamma": 0.5,
  "beta": 0.7,
  "test_broj": 940,
  "test_prosecni_log_dobitak": 0.0009840958297755728,
  "test_priblizni_95_interval": [
    -7.642291006690251e-05,
    0.002044614569618048
  ],
  "model_verovatnoca_NEXT": 7.074675798090557e-08,
  "odnos_prema_uniformnoj": 1.0881514274585558,
  "pojedinacno_ocenjenih_kombinacija": 8485
}





{
  "CSV": "/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_2970.csv",
  "broj_izvlacenja": 2970,
  "poslednja": [
    4,
    10,
    14,
    15,
    28,
    33,
    34
  ],
  "NEXT": [
    2,
    10,
    22,
    29,
    30,
    33,
    39
  ],
  "gamma": 2.0,
  "beta": 0.35,
  "test_broj": 594,
  "test_prosecni_log_dobitak": -0.0010771803702579858,
  "test_priblizni_95_interval": [
    -0.006478526891456988,
    0.0043241661509410174
  ],
  "model_verovatnoca_NEXT": 1.0270143694116122e-07,
  "odnos_prema_uniformnoj": 1.5796443314014734,
  "pojedinacno_ocenjenih_kombinacija": 5712
}





{
  "CSV": "/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_plus_1726.csv",
  "broj_izvlacenja": 1726,
  "poslednja": [
    5,
    6,
    16,
    21,
    23,
    26,
    36
  ],
  "NEXT": [
    2,
    7,
    8,
    11,
    23,
    26,
    34
  ],
  "gamma": 0.0,
  "beta": 0.15,
  "test_broj": 346,
  "test_prosecni_log_dobitak": 2.6382201123937446e-05,
  "test_priblizni_95_interval": [
    -0.00039969723051958326,
    0.0004524616327674582
  ],
  "model_verovatnoca_NEXT": 6.606472258262511e-08,
  "odnos_prema_uniformnoj": 1.016137335965834,
  "pojedinacno_ocenjenih_kombinacija": 5583
}
"""



"""
ako se pronadje skriveni obrazac onda se moze dobiti next kombinacija
ako je pronađeni obrazac stvarna i stabilna zavisnost koja utiče na sledeće izvlačenje 

Sam obrazac u istorijskom CSV-u nije dovoljan: 
može biti slučajnost ili pravilo koje odgovara samo već viđenim rezultatima.

Predmet analize treba da bude zajednička distribucija cele kombinacije i njena promena kroz niz izvlačenja.
Ako sortirano izvlačenje predstavim kao vektor, mogu proučavati:
- Distribuciju položaja i razmaka: gde brojevi leže u dozvoljenom opsegu, koliko su udaljeni i kako se grupišu.
- Zajedničku distribuciju: koje strukture kombinacija nastaju zajedno, uključujući zavisnosti između položaja.
- Distribuciju prelaza: kako struktura jednog izvlačenja prelazi u strukturu narednog.
- Uslovnu distribuciju: da li prethodna izvlačenja menjaju raspodelu mogućih narednih kombinacija.

Moja hipoteza je da u tim distribucijama postoji skrivena pravilnost koja ograničava sledeću kombinaciju. 
To je precizniji cilj od brojanja „čestih“ brojeva.
Ako pronađem takvu pravilnost, proveravam da li na neviđenim izvlačenjima zaista sužava skup mogućih kombinacija. 
Samo ako uslovna distribucija praktično koncentrise svu verovatnoću na jednu kombinaciju, mogu govoriti o određivanju tačne sledeće kombinacije.

Izvorno pitanje je kako grafovske metode primeniti na distribuciju istorijskih kombinacija radi otkrivanja skrivene strukture i eventualnog izvođenja sledeće kombinacije. 

teorija grafova, uz analizu distribucije i promena kroz vreme

39 čvorova, po jedan za svaki broj, i 15.380.937 mogućih hipergrana, po jedna za svaku kombinaciju sedam brojeva.
To je 7-uniformni hipergraf: obična grana povezuje dva čvora, a ovde svaka hipergrana povezuje tačno sedam čvorova.
CSV tada predstavlja vremenski uređen niz realizovanih hipergrana. Analiziram kako su izvučene hipergrane raspoređene u prostoru svih mogućih kombinacija, kako se međusobno preklapaju i kako se taj raspored razvija kroz izvlačenja.
Time zadržavam svih sedam brojeva kao jednu celinu.
U modelu čvor je broj, a jedna grana je kombinacija od sedam čvorova.
Dakle: 39 čvorova i 15.380.937 mogućih grana, a svako izvlačenje realizuje jednu od tih grana. Matematički naziv za takvu granu je hipergrana, jer povezuje sedam čvorova.
Analizu distribucije radim nad tim granama i njihovim redosledom kroz istoriju izvlačenja.
Model sa 39 čvorova i hipergranom za svaku kombinaciju neposredno predstavlja loto 7/39 i čuva sedam brojeva kao celinu.

Moj model, gde je kombinacija čvor, više odgovara kasnijem istraživanju sličnosti i prelaza između kombinacija. On zahteva dodatno definisanje veza, pa rezultat zavisi od toga kako sam izabrao da merim sličnost.
Drugi model je da svaki cvor predstavlja kombinaciju od 7 brojeva. 
Ta dva modela mogu se povezati: od mog hipergrafa mogu napraviti izvedeni graf u kojem hipergrane postaju čvorovi za posebnu analizu. 
Počeo sam mojim hipergrafom, uz vreme i broj pojavljivanja svake izvučene hipergrane. 
Drugi model bih dodao samo ako konkretna analiza to zahteva.
Moj model je bolji izbor za predstavljanje podataka; koji pristup ima prediktivnu vrednost tek treba proveriti na CSV-u.




Izvršio sam prvu analizu mog hipergrafa: 39 čvorova, a svaka kombinacija predstavlja jednu hipergranu. Najizraženije odstupanje pronađeno je u distribuciji preklapanja uzastopnih izvlačenja.
U CSV-u loto7_4696_k79.csv nalazi se 4.696 ispravnih izvlačenja i 4.695 različitih hipergrana. Ista kombinacija 8, 16, 19, 23, 29, 31, 37 pojavljuje se u redovima 2.262 i 4.047.
Za svaka dva uzastopna izvlačenja izračunao sam veličinu preseka njihovih hipergrana:

Zajedničkih čvorova	Zabeleženo prelaza	Očekivano pri nezavisnom, ravnomernom izvlačenju
0	                1.029	            1.027,4
1	                2.001	            1.936,3
2	                1.279	            1.290,9
3	                322	                384,2
4	                57	                53,0
5	                7	                3,2
6	                0	                0,068
7	                0	                0,0003


Kandidat za obrazac jeste manjak prelaza sa tačno tri zajednička čvora — približno 16% manje od očekivanog. Smer odstupanja je isti u starijem i novijem delu podataka.
Proverio sam celu distribuciju preklapanja na vremenskim razmacima od 1 do 50 izvlačenja, uz 1.999 nasumičnih mešanja redosleda. Time kombinacije ostaju iste, a njihove vremenske veze se menjaju:
- Za susedna izvlačenja, posmatrana zasebno: p ≈ 0,0045.
- Kada uračunam da sam pregledao svih 50 razmaka: p ≈ 0,22.
Dakle, postoji konkretno odstupanje za istraživanje, ali ova provera još ne potvrđuje stabilnu vremensku pravilnost.
Napravio sam i početni model koji raspodeljuje verovatnoću naredne hipergrane prema njenom preseku sa prethodnom. Učio je na prvih 3.287 izvlačenja, a proveravan je na preostalih 1.409. Dobitak prema ravnomernoj raspodeli bio je mali, uz interval neizvesnosti koji obuhvata nulu — prediktivna prednost još nije pokazana.
Raspodele raspona kombinacija, uzastopnih nizova i zauzetosti tri jednaka brojčana opsega nisu pokazale izraženo odstupanje u sprovedenim testovima.
Sledeći smislen korak je ispitivanje zajedničke strukture tri uzastopne hipergrane: da li raspored njihovih međusobnih preseka uslovljava naredni prelaz. To može otkriti zavisnost koju analiza samo dva izvlačenja ne vidi.
"""
