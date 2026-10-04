# zahteva NumPy 2.0 ili noviji



import csv
import json
import math

import numpy as np


CSV_FILES = [
    # "/Users/4c/Desktop/GHQ/data/loto7_4696_k79.csv",
    # "/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_2970.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_plus_1726.csv",
]

TOTAL = math.comb(39, 7)

WINDOWS = (1, 2, 3, 4, 5, 7, 10, 16, 32)
GAMMAS = np.array([0.5, 1.0, 2.0, 4.0])
BETAS = np.array([0.15, 0.35, 0.7, 1.2])

MAX_HISTORY = max(WINDOWS)
BEAM_WIDTH = 16
SEARCH_STEPS = 12
BATCH_SIZE = 256

# Broj hipergrana koje sa zadatom hipergranom dele tačno r čvorova.
SHELL_SIZES = np.array([
    math.comb(7, r) * math.comb(32, 7 - r)
    for r in range(8)
])

NORMALIZERS = (
    np.exp(BETAS[:, None] * np.arange(8)) @ SHELL_SIZES
)

# 1 uniformni + 4 statička + 144 vremenska modela.
CONFIGS = (
    [("uniform", 0, 0.0, 0.0)]
    + [("kernel", 0, 0.0, float(beta)) for beta in BETAS]
    + [
        ("temporal", window, float(gamma), float(beta))
        for window in WINDOWS
        for gamma in GAMMAS
        for beta in BETAS
    ]
)


def load_csv(path):
    rows = []

    with open(path, encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row:
                continue

            try:
                values = tuple(sorted(map(int, row)))
            except ValueError as error:
                raise ValueError(
                    f"{path}, red {line_number}: "
                    "očekujem sedam celih brojeva bez zaglavlja."
                ) from error

            if (
                len(values) != 7
                or len(set(values)) != 7
                or min(values) < 1
                or max(values) > 39
            ):
                raise ValueError(
                    f"{path}, red {line_number}: "
                    f"neispravna kombinacija {values}"
                )

            rows.append(values)

    if len(rows) < 200:
        raise ValueError(
            f"{path}: potrebno je najmanje 200 izvlačenja."
        )

    # Jedna maska predstavlja celu hipergranu.
    # Bitovi 0–38 predstavljaju konkretne brojeve 1–39.
    masks = np.array([
        sum(1 << (number - 1) for number in row)
        for row in rows
    ], dtype=np.uint64)

    return rows, masks


def overlap_matrix(masks):
    count = len(masks)
    matrix = np.empty((count, count), dtype=np.uint8)

    for start in range(0, count, BATCH_SIZE):
        matrix[start:start + BATCH_SIZE] = np.bitwise_count(
            masks[start:start + BATCH_SIZE, None]
            & masks[None, :]
        )

    return matrix


def evaluate(matrix, start, stop):
    """
    Prognoza za red t koristi samo ostvarene prelaze j < t.

    Kontekst prelaza j:
        E[j-window], ..., E[j-1]
    Istorijski ishod:
        E[j]

    Sličnost koristi konkretne zajedničke čvorove odgovarajućih
    hipergrana. Skor kandidata zavisi od cele istorijske hipergrane.
    """
    scores = np.zeros((stop - start, len(CONFIGS)))

    for index, t in enumerate(range(start, stop)):
        # Svi modeli koriste isti skup istorijskih ishoda.
        intersections = matrix[t, MAX_HISTORY:t]

        kernels = np.exp(
            BETAS[:, None] * intersections[None, :]
        )

        # Statički modeli: jednaka težina istorijskih ishoda.
        scores[index, 1:5] = np.log(
            kernels.mean(axis=1) / NORMALIZERS * TOTAL
        )

        accumulated = np.zeros(t - MAX_HISTORY)
        weight_sum = 0.0
        column = 5

        for lag in range(1, MAX_HISTORY + 1):
            lag_weight = 1.0 / math.sqrt(lag)
            weight_sum += lag_weight

            accumulated += lag_weight * matrix[
                t - lag,
                MAX_HISTORY - lag:t - lag,
            ]

            if lag not in WINDOWS:
                continue

            context = accumulated / weight_sum

            # Oduzimanje maksimuma održava numeričku stabilnost.
            weights = np.exp(
                (context[:, None] - context.max())
                * GAMMAS[None, :]
            )

            probabilities = (
                (kernels @ weights)
                / weights.sum(axis=0)[None, :]
                / NORMALIZERS[:, None]
            )

            width = len(GAMMAS) * len(BETAS)
            scores[index, column:column + width] = (
                np.log(probabilities * TOTAL).T.ravel()
            )
            column += width

    # Kolona 0 ostaje nula: uniformni model je osnova poređenja.
    return scores


def decode(mask):
    return [
        number + 1
        for number in range(39)
        if int(mask) & (1 << number)
    ]


def predict(masks, config):
    kind, window, gamma, beta = config

    if kind == "uniform":
        # Sve kombinacije imaju istu ocenu.
        # Ovo je deterministički predstavnik, bez prediktivne prednosti.
        return [1, 2, 3, 4, 5, 6, 7], 1.0 / TOTAL, 0

    t = len(masks)
    context = np.zeros(t - MAX_HISTORY)
    weight_sum = 0.0

    for lag in range(1, window + 1):
        lag_weight = 1.0 / math.sqrt(lag)
        weight_sum += lag_weight

        context += lag_weight * np.bitwise_count(
            masks[t - lag]
            & masks[MAX_HISTORY - lag:t - lag]
        )

    if weight_sum:
        context /= weight_sum

    weights = np.exp(gamma * (context - context.max()))
    weights /= weights.sum()

    targets = masks[MAX_HISTORY:]
    kernel = np.exp(beta * np.arange(8))
    normalizer = float(kernel @ SHELL_SIZES)

    cache = {}

    def score_candidates(candidates):
        missing = sorted(
            set(map(int, candidates)) - cache.keys()
        )

        for start in range(0, len(missing), BATCH_SIZE):
            block = missing[start:start + BATCH_SIZE]
            candidate_masks = np.array(block, dtype=np.uint64)

            intersections = np.bitwise_count(
                candidate_masks[:, None] & targets[None, :]
            )

            probabilities = (
                kernel[intersections] @ weights
            ) / normalizer

            cache.update(
                zip(block, map(float, probabilities))
            )

    def leaders():
        # Stabilno razrešenje potpuno jednakih ocena.
        return sorted(
            cache,
            key=lambda mask: (-cache[mask], mask),
        )[:BEAM_WIDTH]

    # Početni skup: konkretne istorijske kombinacije.
    score_candidates(masks)

    # Proširenje: nove kombinacije nastale zamenom jednog čvora.
    # Zadržava se više najboljih kandidata radi šire pretrage.
    for _ in range(SEARCH_STEPS):
        previous_leaders = leaders()
        neighbors = set()

        for mask in previous_leaders:
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
                    neighbors.add(
                        mask ^ (1 << removed) ^ (1 << added)
                    )

        score_candidates(neighbors)

        if leaders() == previous_leaders:
            break

    best = leaders()[0]
    candidate = decode(best)

    assert len(candidate) == len(set(candidate)) == 7

    return candidate, cache[best], len(cache)


def run(path):
    rows, masks = load_csv(path)
    matrix = overlap_matrix(masks)
    count = len(rows)

    # Prvih 60%: početna istorija.
    # Sledećih 20%: izbor modela i parametara.
    # Poslednjih 20%: odvojena provera izabranog modela.
    validation_start = int(count * 0.6)
    test_start = int(count * 0.8)

    metrics = evaluate(matrix, validation_start, count)
    validation_length = test_start - validation_start

    validation_means = metrics[:validation_length].mean(axis=0)

    # Izbor koristi ISKLJUČIVO validaciju, nikada test.
    selected_index = int(np.argmax(validation_means))
    selected = CONFIGS[selected_index]

    test_scores = metrics[
        validation_length:, selected_index
    ]
    test_mean = float(test_scores.mean())

    # NEXT koristi sve trenutno poznate redove tog CSV-a.
    candidate, probability, examined = predict(masks, selected)

    kind, window, gamma, beta = selected

    return {
        "CSV": path,
        "broj_izvlacenja": count,
        "poslednja_kombinacija": list(rows[-1]),
        "NEXT": candidate,
        "broj_uporedjenih_konfiguracija": len(CONFIGS),
        "izabrani_model": kind,
        "duzina_konteksta": window,
        "gamma": gamma,
        "beta": beta,
        "validacija_log_dobitak": float(
            validation_means[selected_index]
        ),
        "test_broj_prognoza": len(test_scores),
        "test_log_dobitak": test_mean,
        "test_opis": (
            "Pozitivan prosečni log-dobitak prema uniformnoj raspodeli."
            if test_mean > 0
            else "Nema pozitivnog prosečnog log-dobitka prema uniformnoj raspodeli."
        ),
        "model_verovatnoca_NEXT": probability,
        "uniformna_verovatnoca": 1.0 / TOTAL,
        "pojedinacno_ocenjenih_kandidata": examined,
        "napomena": (
            "Pretraga je približna. Model-verovatnoća nije "
            "potvrđena stvarna verovatnoća budućeg izvlačenja."
        ),
    }


def main():
    if not hasattr(np, "bitwise_count"):
        raise RuntimeError(
            'Potreban je NumPy >= 2.0. '
            'Instalacija: python3 -m pip install "numpy>=2.0"'
        )

    assert sum(SHELL_SIZES) == TOTAL
    assert len(CONFIGS) == 149

    for path in CSV_FILES:
        result = run(path)
        print(
            json.dumps(result, ensure_ascii=False, indent=2),
            flush=True,
        )


if __name__ == "__main__":
    main()



"""
{
  "CSV": "/Users/4c/Desktop/GHQ/data/loto7_4696_k79.csv",
  "broj_izvlacenja": 4696,
  "poslednja_kombinacija": [
    5,
    6,
    16,
    21,
    23,
    26,
    36
  ],
  "NEXT": [
    10,
    13,
    17,
    26,
    36,
    37,
    39
  ],
  "broj_uporedjenih_konfiguracija": 149,
  "izabrani_model": "temporal",
  "duzina_konteksta": 4,
  "gamma": 2.0,
  "beta": 1.2,
  "validacija_log_dobitak": 0.0007687513733981992,
  "test_broj_prognoza": 940,
  "test_log_dobitak": 0.0034625127022914473,
  "test_opis": "Pozitivan prosečni log-dobitak prema uniformnoj raspodeli.",
  "model_verovatnoca_NEXT": 4.020980528009217e-07,
  "uniformna_verovatnoca": 6.501554489170588e-08,
  "pojedinacno_ocenjenih_kandidata": 9746,
  "napomena": "Pretraga je približna. Model-verovatnoća nije potvrđena stvarna verovatnoća budućeg izvlačenja."
}





{
  "CSV": "/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_2970.csv",
  "broj_izvlacenja": 2970,
  "poslednja_kombinacija": [
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
    6,
    9,
    16,
    22,
    33,
    37
  ],
  "broj_uporedjenih_konfiguracija": 149,
  "izabrani_model": "temporal",
  "duzina_konteksta": 16,
  "gamma": 4.0,
  "beta": 1.2,
  "validacija_log_dobitak": 0.0035278250637462323,
  "test_broj_prognoza": 594,
  "test_log_dobitak": -0.003616856120057807,
  "test_opis": "Nema pozitivnog prosečnog log-dobitka prema uniformnoj raspodeli.",
  "model_verovatnoca_NEXT": 3.0359433644647365e-07,
  "uniformna_verovatnoca": 6.501554489170588e-08,
  "pojedinacno_ocenjenih_kandidata": 6553,
  "napomena": "Pretraga je približna. Model-verovatnoća nije potvrđena stvarna verovatnoća budućeg izvlačenja."
}





{
  "CSV": "/Users/4c/Desktop/GHQ/data/loto7_4696_k79_loto_plus_1726.csv",
  "broj_izvlacenja": 1726,
  "poslednja_kombinacija": [
    5,
    6,
    16,
    21,
    23,
    26,
    36
  ],
  "NEXT": [
    3,
    13,
    19,
    20,
    22,
    28,
    38
  ],
  "broj_uporedjenih_konfiguracija": 149,
  "izabrani_model": "temporal",
  "duzina_konteksta": 5,
  "gamma": 4.0,
  "beta": 0.35,
  "validacija_log_dobitak": 0.00133557001696267,
  "test_broj_prognoza": 346,
  "test_log_dobitak": -0.003816409334653378,
  "test_opis": "Nema pozitivnog prosečnog log-dobitka prema uniformnoj raspodeli.",
  "model_verovatnoca_NEXT": 8.302429384140087e-08,
  "uniformna_verovatnoca": 6.501554489170588e-08,
  "pojedinacno_ocenjenih_kandidata": 7191,
  "napomena": "Pretraga je približna. Model-verovatnoća nije potvrđena stvarna verovatnoća budućeg izvlačenja."
}
"""





"""
V2 je završena i proverena na sva tri CSV-a. 
Poredi 149 konfiguracija, bira parametre na hronološkoj validaciji 
i odvojeno meri rezultat na kasnijim izvlačenjima. 
Nema nasumičnog izbora kombinacije.

CSV	            NEXT	                      Izabrana istorija
4.696 redova	10, 13, 17, 26, 36, 37, 39	  4 izvlačenja
2.970 redova	2, 6, 9, 16, 22, 33, 37	     16 izvlačenja
1.726 redova	3, 13, 19, 20, 22, 28, 38	  5 izvlačenja


Provereni su proračun verovatnoća, odsustvo korišćenja budućih redova i ponovljivost rezultata. 
Ovo je izbor među testiranim modelima, ne dokazano najbolji mogući prediktor. 
Test je pokazao pozitivan dobitak za prvi CSV (najvise kombinacija) i negativan za druga dva. 
Pretraga kandidata nije iscrpna.
"""
