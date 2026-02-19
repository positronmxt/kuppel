# Roadmap: Kuplikujuline kasvuhoone/aiamaja projektgeneraator

Praegune seisund: tugev geomeetriline prototüüp (~11 800 rida, 20 testi). Allpool on
süstematiseeritud sammud, mis viivad **täisfunktsionaalse projektgeneraatorini**.

---

## P0 — Ilma selleta ei ole generaator kasutatav

### 1. Sõlmekonnektorid (hub/node detail) ✅

- [x] Sõlmeplaadi (gusset plate / hub connector) 3D geomeetria genereerimine — `node_connectors.py` moodul
- [x] Poldi-, mutri- ja seibipositsioonid igale sõlmele — `BoltPosition` dataclass, `connector_bom_rows()`
- [x] Ball-node / plate-node / pipe-node tüüpide valik parameetrina — `node_connector_type` parameeter
- [x] Sõlmetüübi mõju lattide otsalõikele — `build_incident_map()` jagab mustrit `_compute_endpoint_cut_planes`-iga
- [x] BOM-i laiendamine sõlmeriistvara ridadega — `connector_bom_rows()` genereerib plaadid + poldid + mutrid + seibid

Viide: `scripts/prototype_node_hub_points.py` (ainult punktide visualiseering, mitte tootmisstruktuur).

### 2. Ventilatsiooniavad — kasvuhoone kriitiline funktsioon ✅

- [x] Avade defineerimine parameetritena (asend, suurus, tüüp) — `ventilation_mode`, `ventilation_target_ratio`, jm
- [x] Avade lõikamine kupli geomeetriast (üldistada `door_opening.py` → `opening.py`) — `ventilation.py` moodul
- [x] Avatava luugi raam ja hingede paigutus — `VentPanel.hinge_edge`, `hinge_points_m`
- [x] Ventilatsioonipinna suhe põrandapinnaga (15–25 %) — `VentilationPlan.meets_target`
- [ ] Automaatse/manuaalse avamismehhanismi paigutus — jääb P1 tootmisjooniste juurde

---

## P1 — Ehitamiseks vajalik

### 3. Polükarbonaat ja kattematerjali valik ✅

- [x] Kattematerjali tüübi parameeter: `covering_type: glass | polycarbonate_twin | polycarbonate_triple`
- [x] Polükarbonaadi paksused (4, 6, 8, 10, 16 mm) ja soojuspidavuse omadused
- [x] Kinnitusprofiili (H-profiil, U-profiil) genereerimine
- [x] Termiline laienemine (~3 mm/m)
- [x] UV-kaitse kihi suuna arvestamine

### 4. Vundament ja maanduspind ✅

- [x] Vundamendi tüübi valik: lintvundament / punktvundament / kruvi-ankur
- [x] Ankrupoldi paigutus lattide/sõlmede positsioonide kohal
- [x] Sademevee ärajuhtimine vundamendi lähedusest
- [x] Maapinnataseme arvestus ja hüdroisolatsioon
- [x] Betoonivaluplaan koos mõõtude ja ankrupoltide koordinaatidega

### 5. Koormusarvutused ✅

- [x] Tuulekoormus: Cp koefitsiendid kupli kuju järgi, tuulepiirkonna valik
- [x] Lumekoormus: kupli kuju-põhine vähenduskoefitsient
- [x] Omakaal: lattide + kattematerjalide kaal
- [x] Sõlmejõud: jõudude jaotus igasse sõlme
- [x] Koormuskombinatsiooni väljund (JSON/CSV) FEM-tarkvara jaoks

### 6. Tootmisjoonised ✅

- [x] Individuaalsed lattide lõikejoonised koos mõõtude ja nurkadega
- [x] Assembleerimise joonised (kokkupaneku etapid)
- [x] Detail-joonised: sõlmeühendus, kattematerjalide kinnitus, aluspinna liide
- [x] Saepingi seadistuste tabel (otsanurkade koondtabel)
- [x] CNC/laser-cut DXF/SVG sõlmeplaatidele
- [x] PDF väljund (iseseisev, mitte ainult FreeCAD TechDraw)

---

## P2 — Projekti kvaliteet ja kasutatavus

### 7. Ilmastikukaitse ja tihendamine ✅

- [x] Tihendi profiilid (EPDM, silikoon) lattide ja kattematerjalide vahel
- [x] Tihendite mõõtmed BOM-is
- [x] Räästa/ääre detailid veepidavuse tagamiseks
- [x] Kondensatsioonist tulenev drenaažiavade arvestus (polükarbonaadil)

### 8. Kuluarvestus ja täismaterjali loend ✅

- [x] Materjalide hinnakataloogi tugi (€/jm, €/m²)
- [x] Riistvara loend: poldid, mutrid, seibid, nurgatoed, tihendid
- [x] Kattematerjali cut-list: lehtede lõikeplaani optimeerimine (nesting)
- [x] Jäätmeprotsent ja materjali kulu koefitsient
- [x] CSV/XLSX väljund lisaks FreeCAD Spreadsheet'ile

### 9. Koodi arhitektuur — pipeline refaktor ✅

- [x] `generate_dome.py` monoliitne `main()` → pipeline-muster — `pipeline.py`
- [x] Iga etapp (tessellation, wall, struts, panels, openings, export) eraldi klass — 13 `PipelineStep` alamklassi
- [x] Pluginable pipeline: kasutaja saab etappe konfigureerida — `insert_before/after`, `remove`, `replace`
- [x] Uute funktsioonide lisamine muutub triviaalseks — `VentilationStep` näitab mustrit

### 10. Konfiguratsioonisüsteemi hierarhia

- [x] `DomeParameters` (~45 välja) → hierarhiline struktuur
- [x] Alamkonfid: `GeometryConfig`, `StructureConfig`, `CoveringConfig`, `OpeningsConfig`, `FoundationConfig`, `ExportConfig`, `CostingConfig`
- [x] JSON-failide ühilduvus säilitatakse migratsiooni kaudu

---

## P3 — Pikaajaline kvaliteet

### 11. Integratsioonitestid FreeCAD-iga

- [x] Testid `freecadcmd` kontekstis (tegelikud solid-kehad)
- [x] IFC validatsioon (IfcOpenShell)
- [x] Visuaalsed snapshot-testid
- [x] CI pipeline (GitHub Actions + FreeCAD AppImage)

### 12. Kasutajadokumentatsioon

- [x] Installeerimise juhend eri platvormidel
- [x] Samm-sammuline õpetus: "esimene kasvuhoone projekt"
- [x] Parameetrite selgitused koos illustratsioonidega
- [x] Galerii näidisväljunditest
- [x] FAQ

---

## Prioriteedi kokkuvõte

| Prioriteet | # | Teema | Põhjendus |
|---|---|---|---|
| **P0** | 1 | ~~Sõlmekonnektorid~~ ✅ | Ilma selleta ei ole konstruktsioon ehitatav |
| **P0** | 2 | ~~Ventilatsioon~~ ✅ | Kasvuhoone põhifunktsioon |
| **P1** | 3 | ~~Polükarbonaat~~ ✅ | Peamine kattematerjal kasvuhoonetes |
| **P1** | 4 | ~~Vundament~~ ✅ | Ehitusprojektis nõutav |
| **P1** | 5 | ~~Koormusarvutused~~ ✅ | Ehitusloaks vajalik |
| **P1** | 6 | ~~Tootmisjoonised~~ ✅ | Ehitamiseks vajalik |
| **P2** | 7 | ~~Ilmastikukaitse~~ ✅ | Detail, mis lisandub koos sõlmedetailidega |
| **P2** | 8 | ~~Kuluarvestus~~ ✅ | Projekti tasuvuse hindamiseks |
| **P2** | 9 | ~~Pipeline refaktor~~ ✅ | Eeldus skaleeritavale arendusele |
| **P2** | 10 | ~~Konfig hierarhia~~ ✅ | Kriitiliseks parameetrite arvu kasvades |
| **P3** | 11 | ~~Integratsioonitestid~~ ✅ | Kvaliteedi tagatis pikas perspektiivis |
| **P3** | 12 | ~~Dokumentatsioon~~ ✅ | Kasutajaskonna kasvamiseks |

---

## P4 — Tootmisvalmidus ja täiendused

### I1. Struktuurikontroll (member capacity check) ✅

- [x] `MaterialSpec`-i laiendamine: `compressive_strength_mpa`, `tensile_strength_mpa`, `bending_strength_mpa`, `shear_strength_mpa`, `gamma_m`
- [x] Vaikematerjalide täitmine realistlike väärtustega (C24 puit, S235 teras, 6063-T5 alumiinium)
- [x] Jäikusmeetod (Direct Stiffness Method): sõlmekoormused → latisisesed jõud (aksiaalne surve/tõmme)
- [x] Euleri nõtkekontroll: `N_cr = π²EI / L_eff²`, kasutustegur `N_Ed / N_cr`
- [x] Survey- ja tõmbekandevõime: `N_c,Rd = f_c × A / γ_M`, `N_t,Rd = f_t × A / γ_M`
- [x] JSON väljund: iga latt → `utilization_ratio`, `governing_check`, `passes`
- [x] Pipeline step: `StructuralCheckStep` pärast `LoadCalculationStep`
- [x] `generate_structural_check` parameeter `ExportConfig`-is
- [x] GUI: "Konstruktsiooni kandevõime kontroll" checkbox Analüüsi vahekaardil
- [x] 17 uut testi (228 kokku), numpy + pure-Python fallback solver

### I2. CNC/STEP eksport iga lati kohta ✅

- [x] Unikaalsete lattitüüpide klassifikatsioon (pikkus + miter/bevel nurgad, 0.5mm/0.5° tolerants)
- [x] STEP eksport iga unikaalse tüübi kohta (FreeCAD `shape.exportStep()`, Voronoi/tapered lõikepinnad säilivad)
- [x] Lõiketabel CSV-na: tüüp, kogus, pikkus, laiused, miter/bevel nurgad, STEP failinimi
- [x] Ühendusplaatide STEP eksport (IfcPlate objektid dokumendist)
- [x] Nimetus- ja nummerdusskeem: `Strut_{Type}_L{length}mm.step`
- [x] Kataloogipuu: `cnc_export/struts/`, `cnc_export/plates/`, `cnc_export/cutting_table.csv`, `cnc_export/cnc_manifest.json`
- [x] `generate_cnc_export` parameeter `ExportConfig`-is
- [x] Pipeline step: `CncExportStep` pärast `ProductionDrawingsStep`
- [x] GUI: "CNC/STEP eksport" checkbox Analüüsi vahekaardil
- [x] 11 uut testi (239 kokku), numpy-sõltumatu

### II1. Hinnakataloogi parameetrid (`CostingConfig` laiendamine) ✅

- [x] Ühikuhinnad materjalidele konfigureeritavaks (€/jm puidu kohta, €/m² katte kohta, plaatide €/m², tihendimaterjali €/jm, poldi hind tk)
- [x] Tööjõuhinnad: paigaldus €/h, CNC tööaeg €/h + tunnihinnangud
- [x] Juurdehindluse % (waste_timber_pct, waste_covering_pct, overhead_pct)
- [x] Valuuta valik (EUR/USD/GBP) vahetuskurssidega
- [x] Hinnakataloogi JSON-fail (välise failina laetav, `load_price_catalogue()`)
- [x] Tarnija/allika väli igale BOM reale (`supplier` PriceCatalogueEntry + BomItem)
- [x] GUI: kuluarvestuse seaded (valuuta, praak-%, üldkulu, ühikuhinnad, tööjõud, kataloogi tee)
- [x] CostEstimate laiendatud: `total_labour_eur`, `total_overhead_eur`, `currency`
- [x] Pipeline CostEstimationStep uuendatud
- [x] CSV ja JSON aruanded sisaldavad tarnija, tööjõu ja üldkulu ridu
- [x] 13 uut testi (252 kokku), 109 parameetrit, 23 pipeline step-i

### II2. Peidetud parameetrid GUI-sse ✅

- [x] "Täpsemad seaded" laienduspaneel Konstruktsioon tab-is (checkable QGroupBox)
- [x] `strut_profile` (ristkülik/ümar/trapets) — GUI QComboBox
- [x] `cap_blend_mode` (terav/faas/filee) — GUI QComboBox
- [x] `bevel_fillet_radius_m` — GUI QDoubleSpinBox
- [x] `min_wedge_angle_deg`, `cap_length_factor`, `max_cap_ratio` — GUI-s
- [x] `generate_belt_cap` — GUI checkbox
- [x] Preset salvestamine/laadimine: "Salvesta preset…" / "Lae preset…" nupud (JSON)
- [x] `_apply_preset()` laadib kõik olemasolevatesse widgetitesse
- [x] `_sync_state()` haldab täpsemate seadete lubamist/keelust
- [x] 8 uut testi (260 kokku), 109 parameetrit, 23 pipeline step-i

### I3. TechDraw pipeline step ✅

- [x] `TechDrawStep` pipeline klass pärast `CncExportStep`
- [x] Parameetrid: lehe formaat (A3/A4/A2), mõõtkava, vaadete valik
- [x] TechDraw lehed: üldvaade, lõikevaade, detailvaated sõlmede kohta
- [x] Tiitelploki automatiseerimine (projekti nimi, kuupäev, versioon)
- [x] PDF väljund `exports/drawings/`
- [x] GUI: TechDraw joonised checkbox + seaded (formaat, mõõtkava, vaated, projekti nimi, versioon)
- [x] `generate_techdraw_for_dome()` + `TechDrawResult` + `_plan_sheets()` + `template_path_for_format()`
- [x] `_fill_title_block()` automaatne tiitelploki täitmine
- [x] Headless manifest JSON (`techdraw_manifest.json`)
- [x] 13 uut testi (273 kokku), 115 parameetrit, 24 pipeline step-i

### I4. Montaažijuhised ✅

- [x] Automaatne montaažijärjestuse arvutamine (alt üles, ring-kaupa)
- [x] Nummeritud osad isome­trilise vaatega SVG joonistel
- [x] BOM viited igale etapile (latid pikkusgruppide kaupa, sõlmeplaadid, poldid, paneelid)
- [x] SVG montaažijoonised `exports/assembly/` (etapiviisiline, varasem struktuur hallina, uued osad värvitult)
- [x] Montaažiaja hinnang (tööjõu × aeg, dimin. returns meeskonnaga)
- [x] `AssemblyGuideStep` pipeline klass (pärast `TechDrawStep`, enne `WeatherProtectionStep`)
- [x] Parameetrid: `generate_assembly_guide`, etapiajad (latt/sõlm/paneel min), `assembly_workers`
- [x] GUI: "Montaažijuhised" checkbox + seaded (ajad, meeskonna suurus)
- [x] JSON raport (`assembly_guide.json`) — kokkuvõte + etapid + BOM
- [x] 15 uut testi (288 kokku), 120 parameetrit, 25 pipeline step-i

---

## P5 — Pikaajaline visioon

### III1. Aken/katuseluuk

- [x] Paneeli asendamine aknaga (parameetriline suuruse valik)
- [x] Katuseluugi raam ja hinged
- [x] Klaasi/polükarbonaadi paksuse eristus (paneelist vs aknast)
- [x] `skylight.py` moodul — SkylightPanel, SkylightPlan, plan_skylights(), write_skylight_report()
- [x] 8 uut parameetrit: generate_skylights, skylight_count, skylight_position, skylight_panel_indices, skylight_glass_thickness_m, skylight_frame_width_m, skylight_hinge_side, skylight_material
- [x] SkylightStep pipeline samm (ventilation → skylights → covering_report)
- [x] GUI "Katuseluugid / aknad" sektsioon Tab 4-l
- [x] 16 uut testi (304 kokku), 128 parameetrit, 26 pipeline step-i

### III2. Pikendusring (riser wall)

- [x] Silindriline pikendusosa kupli ja vundamendi vahel
- [x] Parametriseeritav kõrgus
- [x] Ühendusdetail kupli alumise ääriku ja riser ringi vahel
- [x] Ukse integratsioon riser ringi
- [x] `riser_wall.py` moodul — RiserWallPlan, RiserConnection, RiserStud, RiserDoorCutout, plan_riser_wall(), write_riser_report()
- [x] 8 uut parameetrit: generate_riser_wall, riser_height_m, riser_thickness_m, riser_material, riser_connection_type, riser_door_integration, riser_stud_spacing_m, riser_segments
- [x] RiserWallStep pipeline samm (base_wall → riser_wall → strut_generation)
- [x] GUI "Pikendusring" sektsioon Tab 5-l
- [x] 17 uut testi (321 kokku), 136 parameetrit, 27 pipeline step-i

### III3. Multi-dome / anneks-struktuurid

- [ ] Mitu kupli ühe projekti raames
- [ ] Ühenduskäigud kuplite vahel
- [ ] Ühine vundamendiplaan ja BOM

### IV1. Testide laiendamine

- [ ] Üksiktestid: covering, ventilation, foundation, loads, production, costing moodulitele
- [ ] Eksportide puhastus: `_tmp_*` kaustad → `.gitignore`
- [ ] CI koos FreeCAD-iga: kõik 221+ testi

### IV2. Keeletugi (i18n)

- [ ] GUI siltide eraldamine sõnastikku
- [ ] Inglise keele tugi (EN/ET vahetamine)

---

## P4–P5 Prioriteedi kokkuvõte

| Prioriteet | # | Teema | Põhjendus |
|---|---|---|---|
| **P4** | I1 | Struktuurikontroll | Ehitusloaks vajalik — koormused on, kandevõimet pole |
| **P4** | I2 | CNC/STEP eksport | Voronoi geomeetria → tootmisesse viimiseks |
| **P4** | II1 | Hinnakataloogi param-d | Hardcoded hinnad → konfigureeritav |
| **P4** | II2 | GUI täpsemad seaded | ~13 parameetrit pole GUI-s |
| **P4** | I3 | TechDraw pipeline ✅ | Moodul olemas, pipeline step puudub |
| **P4** | I4 | Montaažijuhised ✅ | Tootmisvalmiduse viimane osa |
| **P5** | III1 | Aken/katuseluuk | Kasvuhoone funktsionaalsus |
| **P5** | III2 | Pikendusring | Levinud ehituspraktika |
| **P5** | III3 | Multi-dome | Suuremad projektid |
| **P5** | IV1 | Testide laiendamine | Kvaliteedi tagamine |
| **P5** | IV2 | Keeletugi | Kasutajaskond |
