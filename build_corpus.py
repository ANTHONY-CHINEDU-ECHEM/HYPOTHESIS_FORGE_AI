import json, random

random.seed(42)

families = [
    ("Garnet LLZO-family", ["Al-doped Li7La3Zr2O12", "Ta-doped Li7La3Zr2O12", "Ga-doped garnet", "Nb-substituted garnet"]),
    ("NASICON-type oxide", ["LAGP", "LATP", "Ge-substituted NASICON", "Sc-doped NASICON"]),
    ("Perovskite-type oxide", ["Li3xLa2/3-xTiO3 (LLTO)", "Sr-doped LLTO", "Nb-doped LLTO"]),
    ("Sulfide argyrodite", ["Li6PS5Cl", "Li6PS5Br", "Sn-substituted argyrodite", "Si-substituted argyrodite"]),
    ("Thio-LISICON phase", ["Li10GeP2S12 (LGPS)", "Li10SnP2S12", "Li9.54Si1.74P1.44S11.7Cl0.3"]),
    ("Lithium halide analogue", ["Li3InCl6", "Li3YCl6", "Li2ZrCl6", "Li3ScCl6"]),
    ("Polymer-in-ceramic composite", ["PEO/LLZO composite", "PVDF-HFP/LATP composite", "Garnet-polymer bilayer"]),
    ("Single-ion conducting polymer", ["Poly(ionic liquid) SPE", "Lithiated Nafion analogue", "Polycarbonate SPE"]),
    ("Amorphous/glassy conductor", ["Li2S-P2S5 glass", "Li2S-SiS2 glass", "Oxysulfide glass-ceramic"]),
]

focus_angles = [
    ("bulk ionic conductivity", "reports room-temperature ionic conductivity measurements as a function of composition, obtained via electrochemical impedance spectroscopy"),
    ("grain-boundary resistance", "isolates grain-boundary versus bulk contributions to total resistance using variable-temperature impedance analysis"),
    ("dopant/substitution chemistry", "examines how aliovalent substitution at a specific crystallographic site changes Li-vacancy concentration and migration energetics"),
    ("interfacial stability with Li metal", "characterizes the chemical and electrochemical stability of the electrolyte/Li-metal interface using XPS and symmetric-cell cycling"),
    ("critical current density / dendrite suppression", "quantifies the critical current density before short-circuit in symmetric Li|electrolyte|Li cells under stepped current cycling"),
    ("processing route effects", "compares sintering or deposition conditions and their effect on relative density, microstructure, and resulting conductivity"),
    ("mechanical properties", "reports fracture toughness and elastic modulus measurements relevant to cell assembly and pressure requirements"),
    ("air/moisture sensitivity", "studies degradation products formed on exposure to humid air and mitigation via surface coatings"),
    ("electrochemical stability window", "determines the oxidative and reductive stability limits via cyclic voltammetry against a Li reference"),
]

verbs = ["This work", "This study", "This report", "This investigation", "This paper"]
findings = [
    "a moderate improvement over the undoped baseline, attributed to increased carrier concentration",
    "a trade-off in which gains in one property come at the cost of another, consistent with prior trends in related systems",
    "no statistically significant change relative to baseline, suggesting the mechanism is not dominant in this composition range",
    "a substantial improvement attributed to microstructural changes accompanying the modification",
    "mixed results depending on processing conditions, highlighting the sensitivity of this class of materials to synthesis route",
    "a previously unreported secondary phase that complicates interpretation of the primary metric",
]

records = []
doc_id = 1
for fam_name, compositions in families:
    for comp in compositions:
        # give each composition 1-2 abstracts on different angles
        angles = random.sample(focus_angles, k=random.choice([1, 2]))
        for angle_name, angle_desc in angles:
            year = random.randint(2015, 2024)
            verb = random.choice(verbs)
            finding = random.choice(findings)
            title = f"{angle_name.capitalize()} in {comp}"
            abstract = (
                f"{verb} {angle_desc} for {comp}, a member of the {fam_name.lower()} family. "
                f"Samples were prepared via standard solid-state or wet-chemical routes appropriate "
                f"to this material class and characterized using complementary structural and "
                f"electrochemical techniques. The results show {finding}. These findings are "
                f"discussed in the context of designing next-generation solid electrolytes for "
                f"lithium-metal batteries, with attention to how {angle_name} relates to overall "
                f"cell performance and manufacturability."
            )
            tags = [fam_name, angle_name]
            records.append({
                "doc_id": f"D{doc_id:04d}",
                "title": title,
                "year": year,
                "tags": tags,
                "abstract": abstract,
            })
            doc_id += 1

print(f"Generated {len(records)} synthetic corpus records")
with open("/home/claude/hypothesisforge/data/corpus/materials_science_corpus.json", "w") as f:
    json.dump({"domain": "solid_state_electrolytes", "records": records}, f, indent=2)
