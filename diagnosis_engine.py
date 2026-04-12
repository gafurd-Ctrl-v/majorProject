"""
diagnosis_engine.py — Subclass-aware diagnosis engine.

With the expanded 27-class model, predictions are now specific subclasses
(e.g. 'IMI', 'CLBBB', 'LVH') rather than coarse superclasses. This means
the engine no longer needs to *infer* territory from attention weights — it
already knows. The role of attention/lead importance shifts to building the
*evidence chain* (which leads the model was watching) rather than guessing
the diagnosis.

Architecture:
  SUBCLASS_CATALOG — static dict: subclass code → clinical metadata
  build_evidence()  — uses lead importance to explain *why* the model fired
  run_diagnosis_engine() — public entry point, unchanged interface

Supports both the new 27-class model and the legacy 5-class model (if
predictions contain 'MI'/'STTC'/'CD'/'HYP'/'NORM' keys, the old dispatcher
is used as a fallback so you don't need to retrain immediately).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from config import CLASSES, LEAD_NAMES, CLASS_NAMES, CLASS_SUPERCLASS


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    lead:       Optional[str]
    factor:     str
    detail:     str
    weight:     float
    supporting: bool


@dataclass
class SubDiagnosis:
    superclass:        str
    name:              str
    confidence:        str
    confidence_score:  float
    territory:         str
    artery:            str
    key_leads:         list[str]
    evidence:          list[EvidenceItem] = field(default_factory=list)
    interpretation:    str               = ''
    differentials:     list[str]         = field(default_factory=list)
    clinical_action:   str               = ''
    urgency:           str               = 'Routine'   # STAT | Urgent | Routine | Info
    combination_notes: list[str]         = field(default_factory=list)


# ── Lead group definitions ─────────────────────────────────────────────────────

INFERIOR_LEADS   = {'II', 'III', 'aVF'}
ANTERIOR_LEADS   = {'V3', 'V4'}
SEPTAL_LEADS     = {'V1', 'V2'}
LATERAL_LEADS    = {'I', 'aVL', 'V5', 'V6'}
HIGH_LAT_LEADS   = {'I', 'aVL'}
LOW_LAT_LEADS    = {'V5', 'V6'}
ANTERO_SEP_LEADS = {'V1', 'V2', 'V3', 'V4'}
POSTERIOR_LEADS  = {'V7', 'V8', 'V9'}   # rarely present; included for completeness
RIGHT_LEADS      = {'V1', 'V2', 'III', 'aVF'}
ALL_PRECORDIAL   = {'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}


def group_importance(lead_importance: dict, group: set) -> float:
    return sum(lead_importance.get(l, 0.0) for l in group)


def top_leads_in_group(lead_importance: dict,
                        group: set,
                        threshold: float = 0.06,
                        fallback_n: int = 2) -> list[tuple[str, float]]:
    """Return leads in *group* above *threshold*, sorted by importance descending.

    If no lead clears the threshold, falls back to the top *fallback_n* leads
    from the group by raw importance so evidence is never silently empty.
    """
    above = sorted(
        [(l, v) for l, v in lead_importance.items() if l in group and v >= threshold],
        key=lambda x: -x[1],
    )
    if above:
        return above
    # Fallback: best available leads in the group, even if below threshold
    fallback = sorted(
        [(l, lead_importance.get(l, 0.0)) for l in group if lead_importance.get(l, 0.0) > 0],
        key=lambda x: -x[1],
    )[:fallback_n]
    return fallback


def top_n_leads(lead_importance: dict, n: int = 3) -> list[str]:
    return [l for l, _ in sorted(lead_importance.items(), key=lambda x: -x[1])[:n]]


# ── Subclass catalog ───────────────────────────────────────────────────────────
#
# Each entry provides the clinical context for a predicted subclass.
# 'key_lead_groups' are sets of lead names. Use set literals — never
# list(set) + list(set), which creates a flat list of strings.
# 'urgency' tiers: STAT (immediate) | Urgent (same-day) | Routine | Info

SUBCLASS_CATALOG: dict[str, dict] = {

    # ── Normal ─────────────────────────────────────────────────────────────────
    'NORM': dict(
        territory='Normal conduction system',
        artery='N/A',
        key_lead_groups=[],
        differentials=['Early / subclinical pathology', 'Normal variant'],
        clinical_action='No immediate action required. Routine follow-up.',
        interpretation='Model classified this recording as within normal limits.',
        urgency='Info',
    ),

    # ── MI subclasses ──────────────────────────────────────────────────────────
    'IMI': dict(
        territory='Inferior wall of left ventricle (diaphragmatic surface)',
        artery='Right Coronary Artery (RCA) — dominant in 85% of patients',
        key_lead_groups=[INFERIOR_LEADS, HIGH_LAT_LEADS],
        differentials=['Posterior MI (check V7–V9)', 'Pericarditis (if diffuse ST elevation)'],
        clinical_action='Urgent cardiology review. Obtain right-sided leads (V4R) to rule out RV involvement.',
        interpretation='Inferior MI pattern. RCA territory. RV involvement occurs in ~40% of inferior MIs.',
        urgency='Urgent',
    ),
    'ASMI': dict(
        territory='Interventricular septum + adjacent anterior wall (V1–V3)',
        artery='Proximal Left Anterior Descending (LAD) — septal perforators',
        key_lead_groups=[ANTERO_SEP_LEADS],
        differentials=['LBBB (can mimic anterior MI)', 'Anterior STEMI (if V4 also involved)'],
        clinical_action='Urgent cardiology review. Proximal LAD — large at-risk territory.',
        interpretation='Anteroseptal MI. Proximal LAD territory. LBBB must be excluded as it can mask or mimic this pattern.',
        urgency='Urgent',
    ),
    'ILMI': dict(
        territory='Inferior + lateral walls (inferolateral)',
        artery='RCA + Left Circumflex (LCx) co-involvement, or dominant LCx',
        key_lead_groups=[INFERIOR_LEADS, LATERAL_LEADS],
        differentials=['Isolated inferior MI + lateral extension', 'LCx dominant territory MI'],
        clinical_action='Urgent cardiology review. Dual-territory involvement — higher haemodynamic risk.',
        interpretation='Inferolateral MI. Combined inferior and lateral wall involvement implicates RCA + LCx or a dominant LCx.',
        urgency='Urgent',
    ),
    'AMI': dict(
        territory='Anterior wall of left ventricle (V3–V5)',
        artery='Left Anterior Descending (LAD) — mid to distal segment',
        key_lead_groups=[ANTERIOR_LEADS, ALL_PRECORDIAL],
        differentials=['Anteroseptal MI', 'Early repolarisation (younger patients)', 'LBBB with Sgarbossa criteria'],
        clinical_action='Urgent cardiology review. LAD territory — highest in-hospital mortality of any MI territory.',
        interpretation='Anterior MI. LAD territory. Largest LV territory; watch for cardiogenic shock and arrhythmia.',
        urgency='Urgent',
    ),
    'ALMI': dict(
        territory='Anterior + lateral walls (anterolateral)',
        artery='Proximal LAD (diagonal branch) or LCx',
        key_lead_groups=[ANTERIOR_LEADS, LATERAL_LEADS],
        differentials=['Anterior MI with lateral extension', 'Proximal LAD occlusion (large territory)'],
        clinical_action='Urgent cardiology review. Combined anterolateral territory — high haemodynamic risk.',
        interpretation='Anterolateral MI. Proximal LAD or LCx. Anterior + lateral wall involvement expands infarct territory.',
        urgency='Urgent',
    ),
    'INJAS': dict(
        territory='Anteroseptal injury / acute ischaemia (V1–V3)',
        artery='Proximal LAD territory (acute)',
        key_lead_groups=[ANTERO_SEP_LEADS],
        differentials=['Evolving STEMI (early stage)', 'Anterior STEMI', 'Brugada pattern'],
        clinical_action='Emergency assessment. Acute anteroseptal injury current — time-critical reperfusion window.',
        interpretation='Anteroseptal injury pattern. Likely acute LAD occlusion with ongoing injury current. Serial ECGs required.',
        urgency='STAT',
    ),
    'LMI': dict(
        territory='Lateral wall of left ventricle (I, aVL, V5–V6)',
        artery='Left Circumflex (LCx) — obtuse marginal branch',
        key_lead_groups=[LATERAL_LEADS],
        differentials=['High lateral MI (I, aVL only)', 'LCx occlusion (often silent / posterior)'],
        clinical_action='Cardiology review. LCx territory. Lateral MIs are often underdiagnosed on standard ECG.',
        interpretation='Lateral MI. LCx territory. The LCx is an electrically "silent" artery — standard 12-lead may under-represent injury extent.',
        urgency='Urgent',
    ),
    'INJAL': dict(
        territory='Anterolateral injury / acute ischaemia',
        artery='Proximal LAD or LCx (diagonal + obtuse marginal)',
        key_lead_groups=[ANTERIOR_LEADS, LATERAL_LEADS],
        differentials=['Evolving STEMI', 'Anterolateral STEMI'],
        clinical_action='Emergency assessment. Acute anterolateral injury — reperfusion window critical.',
        interpretation='Anterolateral injury pattern. Combined anterior and lateral wall ischaemia with ongoing injury current.',
        urgency='STAT',
    ),
    'IPLMI': dict(
        territory='Inferoposterolateral — inferior + posterior + lateral walls',
        artery='Dominant RCA or dominant LCx (large wrap-around)',
        key_lead_groups=[INFERIOR_LEADS, LATERAL_LEADS, SEPTAL_LEADS],
        differentials=['Large RCA territory MI', 'Dominant LCx occlusion'],
        clinical_action='Urgent cardiology review. Large triple-territory infarct — highest haemodynamic risk.',
        interpretation='Inferoposterolateral MI. Three-territory involvement. Dominant vessel (RCA or LCx) supplies all three walls.',
        urgency='Urgent',
    ),
    'IPMI': dict(
        territory='Inferoposterior wall (inferior + posterior)',
        artery='RCA (posterior descending branch) or dominant LCx',
        key_lead_groups=[INFERIOR_LEADS, SEPTAL_LEADS],
        differentials=['Isolated inferior MI', 'Isolated posterior MI'],
        clinical_action='Urgent cardiology review. Posterior involvement suspected — consider V7–V9 leads.',
        interpretation='Inferoposterior MI. Posterior extension of inferior MI. Tall R in V1–V2 (reciprocal) is a key posterior MI sign.',
        urgency='Urgent',
    ),
    'INJIN': dict(
        territory='Inferior injury / acute ischaemia',
        artery='RCA (acute occlusion)',
        key_lead_groups=[INFERIOR_LEADS, HIGH_LAT_LEADS],
        differentials=['Evolving inferior STEMI', 'Pericarditis (if generalised)'],
        clinical_action='Emergency assessment. Acute inferior injury — check right-sided leads for RV involvement.',
        interpretation='Inferior injury pattern. Acute RCA occlusion with ongoing injury current in inferior leads.',
        urgency='STAT',
    ),
    'PMI': dict(
        territory='Posterior wall of left ventricle',
        artery='Posterior Descending Artery (RCA or dominant LCx)',
        key_lead_groups=[SEPTAL_LEADS],
        differentials=['RV hypertrophy (tall R in V1)', 'RBBB (RSR in V1)', 'Posterior STEMI (check V7–V9)'],
        clinical_action='Obtain V7–V9 leads to confirm. Posterior MI is often missed on standard 12-lead ECG.',
        interpretation='Posterior MI. Reciprocal changes in V1–V2 (tall R, ST depression) are the indirect signs on standard ECG.',
        urgency='Urgent',
    ),

    # ── STTC subclasses ────────────────────────────────────────────────────────
    'NDT': dict(
        territory='Ventricular repolarisation (nonspecific)',
        artery='N/A — nonspecific finding',
        key_lead_groups=[ALL_PRECORDIAL, INFERIOR_LEADS, LATERAL_LEADS],
        differentials=['Ischaemia (if new)', 'Electrolyte disturbance', 'Drug effect', 'LVH strain'],
        clinical_action='Compare with prior ECG. If new, investigate: electrolytes, medications, troponin.',
        interpretation='Nonspecific T-wave abnormality. A common, non-diagnostic finding that warrants clinical correlation.',
        urgency='Routine',
    ),
    'NST_': dict(
        territory='Ventricular repolarisation (nonspecific ST)',
        artery='N/A — nonspecific finding',
        key_lead_groups=[ALL_PRECORDIAL, INFERIOR_LEADS],
        differentials=['Ischaemia (if new or dynamic)', 'Early repolarisation (benign)', 'Pericarditis'],
        clinical_action='Compare with prior ECG. If new or symptomatic, urgent cardiology assessment.',
        interpretation='Nonspecific ST changes. Clinical context is essential — static changes are usually benign.',
        urgency='Routine',
    ),
    'DIG': dict(
        territory='Ventricular repolarisation — digitalised pattern',
        artery='N/A — pharmacological effect',
        key_lead_groups=[LOW_LAT_LEADS, INFERIOR_LEADS],
        differentials=['Ischaemia (lateral leads)', 'LVH strain pattern'],
        clinical_action='Correlate with digoxin therapy. Check serum digoxin level and electrolytes.',
        interpretation='Digitalis effect: scooping ST depression ("reverse tick") in lateral leads. Does not indicate toxicity alone.',
        urgency='Routine',
    ),
    'LNGQT': dict(
        territory='Ventricular repolarisation — prolonged QT',
        artery='N/A — repolarisation abnormality',
        key_lead_groups=[INFERIOR_LEADS, LOW_LAT_LEADS],
        differentials=['Drug-induced QT prolongation', 'Electrolyte disturbance (hypoK, hypoMg, hypoCa)', 'Congenital LQTS'],
        clinical_action='Identify and remove QT-prolonging drugs. Correct electrolytes. Genetics referral if congenital suspected.',
        interpretation='Prolonged QT interval (QTc). Risk of Torsades de Pointes. Requires urgent investigation for reversible causes.',
        urgency='Urgent',
    ),
    'ANEUR': dict(
        territory='Anteroapical left ventricle — aneurysm territory',
        artery='LAD (prior infarction with wall motion abnormality)',
        key_lead_groups=[ANTERIOR_LEADS, ANTERO_SEP_LEADS],
        differentials=['Acute anterior STEMI (if new)', 'Anterior MI (subacute phase)'],
        clinical_action='Echocardiogram to confirm aneurysm. Anticoagulation assessment for mural thrombus.',
        interpretation='Ventricular aneurysm pattern. Persistent ST elevation in anterior leads long after MI suggests LV aneurysm.',
        urgency='Urgent',
    ),

    # ── CD subclasses ──────────────────────────────────────────────────────────
    'IRBBB': dict(
        territory='Right bundle branch — partial conduction delay',
        artery='N/A — conduction system',
        key_lead_groups=[SEPTAL_LEADS, INFERIOR_LEADS],
        differentials=['CRBBB', 'Right ventricular hypertrophy', 'Normal variant (especially athletes)'],
        clinical_action='Usually benign. Correlate clinically; investigate if new or symptomatic.',
        interpretation='Incomplete RBBB (QRS 110–119 ms with RSR pattern in V1). Often a normal variant, especially in young athletes.',
        urgency='Info',
    ),
    'CLBBB': dict(
        territory='Left bundle branch — complete conduction block',
        artery='N/A — conduction system (may reflect LAD disease)',
        key_lead_groups=[LOW_LAT_LEADS, ANTERO_SEP_LEADS],
        differentials=['Rate-related LBBB', 'IVCD', 'Anterior MI masked by LBBB (apply Sgarbossa criteria)'],
        clinical_action='Urgent: if new LBBB with chest pain, treat as STEMI equivalent. Cardiology referral.',
        interpretation='Complete LBBB (QRS ≥120ms, negative in V1, broad monophasic R in I/V6). Masks ischaemia — apply Sgarbossa criteria if ACS suspected.',
        urgency='Urgent',
    ),
    'CRBBB': dict(
        territory='Right bundle branch — complete conduction block',
        artery='N/A — conduction system',
        key_lead_groups=[SEPTAL_LEADS, INFERIOR_LEADS],
        differentials=['Posterior MI (tall R in V1)', 'RVH', 'Brugada syndrome'],
        clinical_action='Cardiology review if new. Investigate for underlying right heart disease.',
        interpretation='Complete RBBB (QRS ≥120ms, RSR in V1). Can be idiopathic or indicate right heart disease. Does not obscure ischaemia.',
        urgency='Routine',
    ),
    'IVCD': dict(
        territory='Intraventricular conduction — nonspecific delay',
        artery='N/A — diffuse conduction delay',
        key_lead_groups=[LOW_LAT_LEADS, ANTERO_SEP_LEADS],
        differentials=['LBBB variant', 'RBBB variant', 'Electrolyte abnormality (hyperK)'],
        clinical_action='Correlate clinically. Investigate if new: electrolytes, structural disease.',
        interpretation='Nonspecific IVCD (QRS ≥110ms without typical LBBB or RBBB morphology). Often reflects cardiomyopathy.',
        urgency='Routine',
    ),
    'LAFB': dict(
        territory='Left anterior fascicle — partial left block',
        artery='N/A — anterior fascicle is LAD territory',
        key_lead_groups=[HIGH_LAT_LEADS, INFERIOR_LEADS],
        differentials=['Left axis deviation (other causes)', 'Inferior MI (Q waves vs rS)'],
        clinical_action='Usually benign in isolation. If with RBBB (bifascicular block) or prior syncope, pacing referral.',
        interpretation='Left anterior fascicular block: left axis deviation (−45° to −90°), rS in inferior leads, qR in I/aVL. Bifascicular block if also RBBB.',
        urgency='Routine',
    ),
    'WPW': dict(
        territory='Accessory pathway (Bundle of Kent) — ventricular pre-excitation',
        artery='N/A — accessory conduction pathway',
        # FIX: was list(ALL_PRECORDIAL) + list(INFERIOR_LEADS), which flattened to a
        # list of 9 individual lead *strings*. build_evidence would then call
        # set('V1') → {'V', '1'} for each string "group". Must be a list of sets.
        key_lead_groups=[ALL_PRECORDIAL, INFERIOR_LEADS],
        differentials=['LBBB (negative delta in lateral leads)', 'MI pattern (pseudoinfarction)', 'LVH'],
        clinical_action='Cardiology / electrophysiology referral. Avoid AV-nodal blocking drugs if AF present (risk of VF).',
        interpretation='Wolff-Parkinson-White: delta wave, short PR (<120ms), wide QRS. Risk of supraventricular tachycardia and, if AF, ventricular fibrillation.',
        urgency='Urgent',
    ),

    # ── HYP subclasses ─────────────────────────────────────────────────────────
    'LVH': dict(
        territory='Left ventricle — increased wall thickness / mass',
        artery='N/A — structural diagnosis',
        key_lead_groups=[LOW_LAT_LEADS, SEPTAL_LEADS],
        differentials=['Athletic heart (high voltage, no strain)', 'HCM (hypertrophic cardiomyopathy)', 'Aortic stenosis'],
        clinical_action='Echocardiogram to confirm. Investigate for hypertension, aortic stenosis, HCM.',
        interpretation='LVH: Sokolow-Lyon criterion S(V1)+R(V5/V6)>35mm, or Cornell >28mm(M)/20mm(F). Strain pattern in lateral leads if present.',
        urgency='Routine',
    ),
    'RVH': dict(
        territory='Right ventricle — increased wall thickness / pressure load',
        artery='N/A — structural diagnosis',
        key_lead_groups=[SEPTAL_LEADS, INFERIOR_LEADS],
        differentials=['Posterior MI (tall R in V1)', 'RBBB (mimics dominant R in V1)', 'Normal variant in children'],
        clinical_action='Echocardiogram and pulmonary function tests. Investigate for pulmonary hypertension, COPD, congenital heart disease.',
        interpretation='RVH: dominant R in V1 (R>S), right axis deviation. Indicates raised RV pressure or volume load.',
        urgency='Routine',
    ),
    'SEHYP': dict(
        territory='Subendocardial — pressure/volume overload strain pattern',
        artery='N/A — structural or haemodynamic',
        key_lead_groups=[LOW_LAT_LEADS, INFERIOR_LEADS],
        differentials=['LVH strain', 'Lateral ischaemia / NSTEMI', 'Digoxin effect'],
        clinical_action='Echocardiogram to assess for structural disease. Rule out ischaemia with troponin.',
        interpretation='Subendocardial hypertrophy / strain: downsloping ST depression with T-wave inversion, typically lateral. Reflects pressure overload.',
        urgency='Routine',
    ),
}


# ── Combination rules ──────────────────────────────────────────────────────────
#
# Each rule: (frozenset of class codes that must ALL be present) → note string.
# Evaluated after all individual diagnoses are built; matched notes are appended
# to every SubDiagnosis in the triggering set.

_COMBINATION_RULES: list[tuple[frozenset[str], str, str]] = [
    # (trigger set,  note,  urgency upgrade if 'STAT'|'Urgent'|None)

    # Acute STEMI equivalents
    (frozenset({'IMI', 'INJIN'}),
     'Co-occurring inferior MI + inferior injury pattern — high probability of acute inferior STEMI. '
     'Obtain right-sided leads (V4R) immediately to assess RV involvement.',
     'STAT'),
    (frozenset({'ASMI', 'INJAS'}),
     'Anteroseptal infarction with active injury current — consistent with acute LAD occlusion (proximal). '
     'Time-critical reperfusion window.',
     'STAT'),
    (frozenset({'AMI', 'INJAS'}),
     'Anterior MI with ongoing anteroseptal injury — acute LAD STEMI pattern. Large LV territory at risk.',
     'STAT'),
    (frozenset({'ALMI', 'INJAL'}),
     'Anterolateral MI with active anterolateral injury current — acute large-territory STEMI. '
     'Proximal LAD or LCx culprit.',
     'STAT'),
    (frozenset({'ILMI', 'INJIN'}),
     'Inferolateral MI with inferior injury — large two-territory STEMI. Consider dominant LCx or RCA culprit.',
     'STAT'),
    (frozenset({'IMI', 'IPMI'}),
     'Combined inferior + posterior MI — large RCA territory. Reciprocal changes in V1–V2 confirm posterior extension.',
     'Urgent'),
    (frozenset({'IPLMI', 'INJIN'}),
     'Triple-territory inferoposterolateral MI with active injury — extremely large STEMI. '
     'Immediate haemodynamic support may be required.',
     'STAT'),

    # LBBB masking ischaemia
    (frozenset({'CLBBB', 'IMI'}),
     'LBBB co-present with inferior MI — LBBB can mask inferior ischaemia. Apply modified Sgarbossa criteria. '
     'New LBBB with chest pain = STEMI equivalent.',
     'STAT'),
    (frozenset({'CLBBB', 'AMI'}),
     'LBBB co-present with anterior MI — LBBB obscures anterior ischaemia. Apply Sgarbossa criteria. '
     'Treat as STEMI equivalent if new.',
     'STAT'),
    (frozenset({'CLBBB', 'ASMI'}),
     'LBBB with anteroseptal MI pattern — ischaemia is likely masked. Sgarbossa criteria must be applied.',
     'STAT'),

    # Bifascicular block → pacing risk
    (frozenset({'LAFB', 'CRBBB'}),
     'Bifascicular block (LAFB + CRBBB) — significant risk of progression to complete AV block. '
     'Pacing referral required if symptomatic or new.',
     'Urgent'),
    (frozenset({'LAFB', 'CLBBB'}),
     'LAFB + CLBBB pattern — possible trifascicular involvement. High-degree AV block risk. '
     'Urgent electrophysiology review.',
     'Urgent'),

    # Arrhythmia risk combinations
    (frozenset({'LNGQT', 'WPW'}),
     'Prolonged QT interval with WPW pre-excitation — extreme arrhythmia risk. '
     'Risk of TdP and antidromic AF degenerating to VF. Immediate electrophysiology referral.',
     'STAT'),

    # Hypertrophy + strain
    (frozenset({'LVH', 'SEHYP'}),
     'LVH with subendocardial strain pattern — advanced hypertensive heart disease or hypertrophic cardiomyopathy. '
     'Echocardiogram mandatory to differentiate HCM from hypertensive LVH.',
     'Urgent'),
    (frozenset({'RVH', 'CRBBB'}),
     'RVH with complete RBBB — consistent with significant right heart pressure overload (pulmonary hypertension, '
     'cor pulmonale). Echocardiogram + BNP.',
     'Urgent'),

    # Posterior MI + RBBB ambiguity
    (frozenset({'PMI', 'CRBBB'}),
     'Posterior MI pattern and RBBB both present — RBBB causes dominant R in V1, which can mimic posterior MI. '
     'V7–V9 leads and echocardiogram required to differentiate.',
     'Urgent'),

    # High-voltage + conduction
    (frozenset({'LVH', 'CLBBB'}),
     'LVH with complete LBBB — likely hypertensive or dilated cardiomyopathy. '
     'Ischaemia completely masked; apply Sgarbossa criteria if ACS suspected.',
     'Urgent'),
]


# ── Evidence builder ───────────────────────────────────────────────────────────

def _lead_territory_note(lead: str, subclass: str) -> str:
    """Return a concise clinical note for why this lead matters for this diagnosis."""
    notes = {
        ('II',   'IMI'):   'Inferior lead — directly faces inferior LV wall',
        ('III',  'IMI'):   'Inferior lead — key for RCA territory',
        ('aVF',  'IMI'):   'Inferior lead — diaphragmatic surface of LV',
        ('V3',   'AMI'):   'Anterior precordial — faces LAD territory',
        ('V4',   'AMI'):   'Anterior precordial — LAD mid territory',
        ('V1',   'ASMI'):  'Septal lead — first septal perforator territory',
        ('V2',   'ASMI'):  'Septal lead — proximal LAD territory',
        ('I',    'ALMI'):  'Lateral lead — high lateral wall (diagonal branch)',
        ('aVL',  'ALMI'):  'High lateral — reciprocal in inferior MI; primary in lateral MI',
        ('V5',   'LVH'):   'Lateral precordial — tall R wave, Sokolow-Lyon criterion',
        ('V6',   'LVH'):   'Lateral precordial — LVH voltage criterion',
        ('V1',   'CRBBB'): 'Septal — RSR prime in RBBB; right-sided delay',
        ('V1',   'CLBBB'): 'Septal — broad negative complex in LBBB',
        ('V5',   'CLBBB'): 'Lateral — broad monophasic R wave in LBBB',
        ('V1',   'RVH'):   'Dominant R in V1 — hallmark of right ventricular hypertrophy',
        ('V1',   'PMI'):   'Reciprocal change (tall R, ST depression) in posterior MI',
        ('V2',   'PMI'):   'Reciprocal change in posterior MI',
        ('II',   'INJIN'): 'Inferior lead — injury current in acute RCA occlusion',
        ('aVF',  'INJIN'): 'Inferior lead — diaphragmatic ischaemia pattern',
        ('V1',   'WPW'):   'Pre-excitation most visible in septal / right precordial leads',
        ('V2',   'WPW'):   'Delta wave morphology in septal lead',
        ('I',    'WPW'):   'Delta wave polarity determines pathway location',
        ('V5',   'SEHYP'): 'Lateral ST-T changes — subendocardial strain',
        ('V6',   'SEHYP'): 'Lateral ST-T changes — pressure overload pattern',
        ('V1',   'ANEUR'): 'Persistent ST elevation — anteroseptal aneurysm territory',
        ('V2',   'INJAS'): 'Active injury current — proximal LAD occlusion',
        ('V3',   'INJAS'): 'Active injury current — anteroseptal zone',
    }
    return notes.get((lead, subclass), f'Relevant territory for {subclass} pattern')


def build_evidence(subclass:        str,
                   lead_importance: dict,
                   model_prob:      float,
                   metrics:         dict) -> list[EvidenceItem]:
    """Build evidence chain from lead importance and clinical metrics.

    The subclass tells us *what* was found; lead importance explains *why*
    the model was confident (which leads drove the prediction).
    """
    info     = SUBCLASS_CATALOG.get(subclass, {})
    evidence = []

    # Lead-level evidence — pull from each key lead group.
    # top_leads_in_group now falls back to best available leads when none
    # clear the primary threshold, so evidence is never silently empty.
    groups_seen: set[frozenset] = set()
    for group in info.get('key_lead_groups', []):
        # Normalise to set — guard against any non-set type in the catalog
        if not isinstance(group, (set, frozenset)):
            group = set(group)
        key = frozenset(group)
        if key in groups_seen:
            continue
        groups_seen.add(key)
        for lead, imp in top_leads_in_group(lead_importance, group, threshold=0.06):
            territory_note = _lead_territory_note(lead, subclass)
            evidence.append(EvidenceItem(
                lead=lead,
                factor=f'Model attention: {imp * 100:.1f}%',
                detail=territory_note,
                weight=imp,
                supporting=True,
            ))

    # Metric-level evidence
    hr  = metrics.get('hr')
    pr  = metrics.get('pr_interval')
    qrs = metrics.get('qrs_duration')
    qtc = metrics.get('qtc')
    sup = CLASS_SUPERCLASS.get(subclass, subclass)

    if qrs is not None and sup == 'CD':
        if qrs > 120:
            evidence.append(EvidenceItem(None, f'Wide QRS: {qrs} ms',
                                         'Conduction delay criterion (≥120 ms)', 0.8, True))
        elif qrs > 110:
            evidence.append(EvidenceItem(None, f'Borderline QRS: {qrs} ms',
                                         'Borderline conduction slowing (110–119 ms)', 0.5, True))

    if qtc is not None and subclass == 'LNGQT':
        if qtc > 500:
            level = 'severely prolonged — high TdP risk'
        elif qtc > 450:
            level = 'prolonged'
        else:
            level = 'borderline'
        evidence.append(EvidenceItem(None, f'QTc: {qtc} ms ({level})',
                                     'QTc measurement supports LQTS classification', 0.9, True))

    if hr is not None and hr > 100 and sup in ('STTC', 'NORM'):
        evidence.append(EvidenceItem(None, f'Tachycardia: {hr} bpm',
                                     'Rate-related repolarisation changes are possible', 0.4, False))

    if hr is not None and hr < 50 and subclass == 'WPW':
        evidence.append(EvidenceItem(None, f'Bradycardia: {hr} bpm',
                                     'Atypical for WPW — assess for high-degree block', 0.3, False))

    if pr is not None and pr < 120 and subclass == 'WPW':
        evidence.append(EvidenceItem(None, f'Short PR: {pr} ms',
                                     'Short PR is a cardinal WPW feature (accessory pathway bypass)', 0.85, True))

    return evidence


# ── NORM special handler ───────────────────────────────────────────────────────

def _build_norm_diagnosis(lead_importance: dict,
                           metrics:          dict,
                           model_prob:       float) -> SubDiagnosis:
    """NORM gets a more nuanced treatment using clinical metrics.

    NOTE: the `flags` list captures *all* metric deviations first. The subsequent
    name / interpretation / action selection checks flags to avoid dead-code
    branches that could never be reached after the flag-collection step.
    """
    hr  = metrics.get('hr')
    pr  = metrics.get('pr_interval')
    qrs = metrics.get('qrs_duration')
    qtc = metrics.get('qtc')

    flags: list[str] = []
    if hr  and hr  > 100:  flags.append(f'HR {hr} bpm — sinus tachycardia')
    if hr  and hr  <  60:  flags.append(f'HR {hr} bpm — sinus bradycardia')
    if pr  and pr  > 200:  flags.append(f'PR {pr} ms — borderline 1st-degree AV block')
    if qrs and qrs > 100:  flags.append(f'QRS {qrs} ms — borderline conduction slowing')
    if qtc and qtc > 450:  flags.append(f'QTc {qtc} ms — borderline prolongation')

    # Derive name and action from flags (previously the elif hr/hr branches were
    # dead code because their conditions were already captured in flags above)
    if not flags:
        name   = 'Normal Sinus Rhythm'
        interp = 'All intervals within normal limits. ECG appearance within normal range.'
        action = 'No immediate action required. Routine follow-up.'
    elif len(flags) == 1 and 'tachycardia' in flags[0]:
        name   = 'Sinus Tachycardia'
        interp = f'Heart rate {hr} bpm — sinus tachycardia. Model otherwise normal.'
        action = 'Identify cause: pain, anxiety, fever, hypovolaemia, PE, medications.'
    elif len(flags) == 1 and 'bradycardia' in flags[0]:
        name   = 'Sinus Bradycardia'
        interp = f'Heart rate {hr} bpm — sinus bradycardia. Can be physiological in athletes.'
        action = 'Evaluate if symptomatic. Check beta-blockers and other rate-lowering drugs.'
    else:
        name   = 'Normal with minor metric variants'
        interp = f'ECG classified as normal by model with minor metric flags: {"; ".join(flags)}.'
        action = 'No immediate action. Correlate clinically with flagged findings.'

    evidence = (
        [EvidenceItem(None, flag, 'Clinical metric', 0.7, True) for flag in flags]
        or
        [EvidenceItem(None, 'All intervals normal',
                      f'HR: {hr} bpm  PR: {pr} ms  QRS: {qrs} ms  QTc: {qtc} ms',
                      0.9, True)]
    )

    return SubDiagnosis(
        superclass='NORM',
        name=name,
        confidence='High' if model_prob > 0.75 else 'Moderate',
        confidence_score=model_prob,
        territory=SUBCLASS_CATALOG['NORM']['territory'],
        artery='N/A',
        key_leads=top_n_leads(lead_importance, 2),
        evidence=evidence,
        interpretation=interp,
        differentials=SUBCLASS_CATALOG['NORM']['differentials'],
        clinical_action=action,
        urgency='Info' if not flags else 'Routine',
    )


# ── Combination detection ──────────────────────────────────────────────────────

def _apply_combination_rules(results: list[SubDiagnosis]) -> None:
    """Mutate *results* in-place: append cross-diagnosis notes and upgrade urgency.

    Only rules whose entire trigger set is present in the result list fire.
    The urgency of every SubDiagnosis in the triggering set is upgraded if the
    rule specifies a higher urgency level.
    """
    _urgency_rank = {'Info': 0, 'Routine': 1, 'Urgent': 2, 'STAT': 3}
    code_to_diag: dict[str, SubDiagnosis] = {
        d.name: d for d in results  # keyed by name is less reliable; use cls code
    }
    # Build a code → SubDiagnosis map using CLASSES (which maps position → code)
    cls_set = {CLASS_NAMES.get(cls, cls): cls for cls in CLASSES}  # name → code
    code_map: dict[str, SubDiagnosis] = {}
    for d in results:
        # Try to reverse-map name back to class code
        for code in CLASSES:
            if CLASS_NAMES.get(code, code) == d.name or code == d.name:
                code_map[code] = d
                break

    present_codes = set(code_map.keys())

    for trigger, note, urgency_upgrade in _COMBINATION_RULES:
        if not trigger.issubset(present_codes):
            continue
        for code in trigger:
            diag = code_map[code]
            if note not in diag.combination_notes:
                diag.combination_notes.append(note)
            # Upgrade urgency if rule demands it
            if urgency_upgrade and (
                _urgency_rank.get(urgency_upgrade, 0) > _urgency_rank.get(diag.urgency, 0)
            ):
                diag.urgency = urgency_upgrade


# ── Main engine ────────────────────────────────────────────────────────────────

# Minimum probability to report a class (a secondary filter applied on top of
# the per-class thresholds in app.py — keeps the engine output focused)
THRESHOLD: float = 0.30


def run_diagnosis_engine(predictions:     dict,
                          lead_importance: dict,
                          metrics:         dict) -> list[SubDiagnosis]:
    """Run the full diagnosis engine for all classes above THRESHOLD.

    Works with both the new 27-class model and the legacy 5-class model.

    Args:
        predictions:     {class_code: probability}  e.g. {'IMI': 0.92, 'INJIN': 0.44}
        lead_importance: {lead_name: fraction}       e.g. {'II': 0.22, 'III': 0.18}
        metrics:         dict from report.extract_clinical_metrics()

    Returns:
        List of SubDiagnosis sorted by confidence_score descending, with
        cross-diagnosis combination notes appended where applicable.
    """
    _metrics: dict = metrics or {}
    _lead_imp: dict = lead_importance or {}
    results: list[SubDiagnosis] = []

    for cls, prob in predictions.items():
        if prob < THRESHOLD:
            continue

        # NORM has a specialised metric-aware handler
        if cls == 'NORM':
            results.append(_build_norm_diagnosis(_lead_imp, _metrics, prob))
            continue

        # New 27-class subclass — direct lookup
        if cls in SUBCLASS_CATALOG:
            info       = SUBCLASS_CATALOG[cls]
            conf_score = float(prob)
            confidence = (
                'High'     if conf_score >= 0.70 else
                'Moderate' if conf_score >= 0.45 else
                'Low'
            )
            evidence = build_evidence(cls, _lead_imp, conf_score, _metrics)

            # Collect key leads from all lead groups in the catalog entry
            all_groups: set[str] = set()
            for g in info['key_lead_groups']:
                if not isinstance(g, (set, frozenset)):
                    g = set(g)
                all_groups.update(g)
            key_leads = [
                l for l, _ in top_leads_in_group(_lead_imp, all_groups or set(LEAD_NAMES), 0.04)
            ][:5]
            if not key_leads:
                key_leads = top_n_leads(_lead_imp, 3)

            results.append(SubDiagnosis(
                superclass=CLASS_SUPERCLASS.get(cls, cls),
                name=CLASS_NAMES.get(cls, cls),
                confidence=confidence,
                confidence_score=conf_score,
                territory=info['territory'],
                artery=info['artery'],
                key_leads=key_leads,
                evidence=evidence,
                interpretation=info['interpretation'],
                differentials=info['differentials'],
                clinical_action=info['clinical_action'],
                urgency=info.get('urgency', 'Routine'),
            ))

        # Legacy 5-class fallback (if still running the old model)
        elif cls in ('MI', 'STTC', 'CD', 'HYP'):
            results.append(SubDiagnosis(
                superclass=cls,
                name=f'{cls} (superclass — retrain for specific subclass)',
                confidence='Low',
                confidence_score=prob * 0.5,
                territory='Unknown — retrain model for subclass predictions',
                artery='Unknown',
                key_leads=top_n_leads(_lead_imp, 3),
                interpretation=(
                    f'Legacy 5-class model detected {cls}. '
                    'Retrain with the updated config.py to get specific subclass diagnoses.'
                ),
                differentials=[],
                clinical_action='Retrain the model with the expanded 27-class config to get actionable subclass predictions.',
                urgency='Routine',
            ))

    # Sort by confidence descending, then apply cross-diagnosis rules
    results.sort(key=lambda x: -x.confidence_score)
    _apply_combination_rules(results)

    return results