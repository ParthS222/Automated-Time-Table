import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import hashlib
from collections import defaultdict
import numpy as np
import random
import re
import io # Added for in-memory CSV creation

# -------------------------
# CONFIG
# -------------------------
LECTURE_MIN = 90
TUTORIAL_MIN = 60
LAB_MIN = 120
DAY_FIRST_MIN = 9 * 60
CLASSES_END_MIN = 18 * 60 + 30
DAY_FULL_END_MIN = 20 * 60 + 30
LUNCH_START_MIN = 12 * 60 + 30
LUNCH_DURATION_MIN = 90
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
TIME_UNIT_MIN = 15
BREAK_MIN = 15
BREAK_UNITS = BREAK_MIN // TIME_UNIT_MIN

# The big halls list you confirmed (these will be excluded for EXAM TIMETABLE + SEATING)
BIG_HALLS_TO_EXCLUDE_FOR_EXAMS = {"C004", "C002", "C003", "C501", "C502", "C503"}

# Default exclude rooms (keeps default behaviour for seating UI); we will use BIG_HALLS_TO_EXCLUDE_FOR_EXAMS here
DEFAULT_EXCLUDE_ROOMS = set(BIG_HALLS_TO_EXCLUDE_FOR_EXAMS)
# MAX_EXAMS_PER_SLOT constant defined here for use in the UI and scheduler
MAX_EXAMS_PER_SLOT_DEFAULT = 4 

# -------------------------
# UTILITIES
# -------------------------
def minutes_to_time(m: int) -> str:
    h = m // 60
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

def generate_color_from_string(s: str):
    # Format: (Uniform Dark Grey Background, Vibrant Accent Border)
    palette = [
        ("#222222", "#FF5F56"),  # Neon Red
        ("#222222", "#42CC8C"),  # Emerald Green
        ("#222222", "#FFB829"),  # Golden Yellow
        ("#222222", "#5D9CE6"),  # Ocean Blue
        ("#222222", "#C965DF"),  # Magenta
        ("#222222", "#28C8C6"),  # Bright Cyan
        ("#222222", "#FF8855"),  # Bright Orange
        ("#222222", "#B877FF"),  # Light Purple
    ]
    h = int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
    return palette[h % len(palette)]
    

def _normalize_instructors(instr_str):
    if pd.isna(instr_str) or instr_str is None or str(instr_str).strip() == '':
        return []
    for sep in [';', ',', '&', ' and ']:
        if sep in instr_str:
            return [p.strip() for p in instr_str.split(sep) if p.strip()]
    return [instr_str.strip()]

# -------------------------
# Normalize combined-group rows (keep combined rows, record components)
# -------------------------
def normalize_combined_groups(all_courses_df, show_log=True):
    df = all_courses_df.copy().reset_index(drop=True)

    def clean_group(g):
        if pd.isna(g):
            return ""
        s = str(g).strip()
        if ";" in s:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            return ";".join(parts)
        return s

    df['group'] = df['group'].apply(clean_group)

    def components_from_group(g):
        if not g:
            return ['COMMON']
        if ";" in g:
            return [p.strip() for p in g.split(";") if p.strip()]
        return [g]

    def canonical_group_id(g):
        comps = components_from_group(g)
        if len(comps) == 1:
            return comps[0]
        safe = [c.replace(" ", "_").replace("-", "_") for c in comps]
        return "_".join(safe)

    df['group_components'] = df['group'].apply(components_from_group)
    df['group_id'] = df['group'].apply(canonical_group_id)

    if show_log:
        combined_rows = df[df['group'].str.contains(';', na=False)]
        if not combined_rows.empty:
            st.info("Found combined group rows (kept as single sessions). They will block all component groups:")
            for _, r in combined_rows.iterrows():
                st.write(f"- {r.get('code','')} (Y{int(r.get('year',1))}): groups -> {r['group_components']}, students={int(r.get('students',0))}")
    return df

# -------------------------
# Finalize groups cleanup (drop redundant combined-rows)
# -------------------------
def _finalize_group_rows(df):
    """
    If a row has group like "CSE-A;CSE-B" but the SAME course+year already has
    per-component rows for CSE-A and CSE-B, drop the combined row to avoid
    creating an extra 'combined' batch in the timetable.
    """
    df = df.reset_index(drop=True).copy()
    to_drop = set()

    for idx, row in df[df['group'].astype(str).str.contains(';', na=False)].iterrows():
        code = str(row.get('code', '')).strip()
        try:
            year = int(row.get('year', 1))
        except Exception:
            year = 1
        comps = [p.strip() for p in str(row['group']).split(';') if p.strip()]
        if not comps:
            continue

        # Check if per-component rows exist for ALL components
        all_present = True
        for comp in comps:
            present = not df[
                (df['code'].astype(str).str.strip() == code) &
                (df['year'].astype(int) == year) &
                (df['group'].astype(str).str.strip() == comp)
            ].empty
            if not present:
                all_present = False
                break

        if all_present:
            to_drop.add(idx)

    if to_drop:
        df = df.drop(index=list(to_drop)).reset_index(drop=True)
    return df

# -------------------------
# Roll number generator
# -------------------------
from collections import defaultdict
def generate_roll_number_v2(academic_year: int, section_or_group: str, branch_serial_map: dict):
    year_prefix_map = {1: 25, 2: 24, 3: 23, 4: 22}
    prefix = str(year_prefix_map.get(academic_year, 'XX'))
    branch_key = str(section_or_group).upper().split('-')[0].split(';')[0].split()[0]
    branch_map = {'CSE': 'CS', 'ECE': 'EC', 'DSAI': 'DS', 'COMMON': 'XX'}
    branch_code = branch_map.get(branch_key, 'XX')
    key = (academic_year, section_or_group)
    branch_serial_map[key] += 1
    serial = branch_serial_map[key]
    return f"{prefix}B{branch_code}{serial:03d}"

# -------------------------
# TIMETABLE (CP-SAT) using group
# -------------------------
def decompose_sessions(all_courses):
    sessions = []
    for idx, c in enumerate(all_courses):
        L_count = int(c.get('L', 0) or 0)
        T_count = int(c.get('T', 0) or 0)
        P_count = int(c.get('P', 0) or 0)
        
        # NEW: Extract Basket ID from the elective column
        elec_val = str(c.get('elective', '')).strip().upper()
        is_elective = elec_val not in ['', 'FALSE', '0', 'NO', 'NAN']
        basket = '1' # Default to basket 1 if they just put "TRUE"
        if is_elective:
            match = re.search(r'\d+', elec_val)
            if match:
                basket = match.group(0)
                
        group = str(c.get('group', c.get('section', 'COMMON')))
        
        for occ in range(L_count):
            sessions.append({'course_idx': idx, 'activity': 'L', 'duration_min': LECTURE_MIN, 'elective': is_elective, 'basket': basket, 'group': group, 'occurrence': occ})
        for occ in range(T_count):
            sessions.append({'course_idx': idx, 'activity': 'T', 'duration_min': TUTORIAL_MIN, 'elective': is_elective, 'basket': basket, 'group': group, 'occurrence': occ})
        for occ in range(P_count):
            sessions.append({'course_idx': idx, 'activity': 'P', 'duration_min': LAB_MIN, 'elective': is_elective, 'basket': basket, 'group': group, 'occurrence': occ})
    return sessions

# ----- REPLACE the existing generate_timetable_fast with this version -----

def _preflight_checks(all_courses, rooms):
    """Return list of human-readable problems (empty list = ok).
    Also return a small diagnostics dict for printing helpful info.
    """
    problems = []
    diag = {'num_courses': len(all_courses), 'num_rooms': len(rooms)}

    # room capacities
    room_caps = []
    lab_room_caps = []
    for r in rooms:
        try:
            rc = int(r.get('capacity', 0) or 0)
        except Exception:
            rc = 0
        room_caps.append(rc)
        if str(r.get('name','')).upper().startswith('L'):
            lab_room_caps.append(rc)
    max_room_cap = max(room_caps) if room_caps else 0
    total_lab_cap = sum(lab_room_caps) if lab_room_caps else 0
    diag['max_room_capacity'] = max_room_cap
    diag['total_lab_capacity'] = total_lab_cap

    # minutes & units (same as solver)
    minutes_list = [m for m in range(DAY_FIRST_MIN, DAY_FULL_END_MIN, TIME_UNIT_MIN)
                    if not (LUNCH_START_MIN <= m < LUNCH_START_MIN + LUNCH_DURATION_MIN)]
    units_per_day = len(minutes_list)
    units_for_core_classes = len([m for m in minutes_list if m < CLASSES_END_MIN])
    total_core_week_units = units_for_core_classes * len(DAYS)
    total_week_units = units_per_day * len(DAYS)
    diag['units_per_day'] = units_per_day
    diag['core_week_units'] = total_core_week_units
    diag['week_units'] = total_week_units

    # Build sessions (like solver does)
    sessions = []
    for idx, c in enumerate(all_courses):
        students = int(c.get('students', 0) or 0)
        L_count = int(c.get('L', 0) or 0)
        T_count = int(c.get('T', 0) or 0)
        P_count = int(c.get('P', 0) or 0)
        for _ in range(L_count):
            sessions.append((idx, 'L', LECTURE_MIN, students))
        for _ in range(T_count):
            sessions.append((idx, 'T', TUTORIAL_MIN, students))
        for _ in range(P_count):
            sessions.append((idx, 'P', LAB_MIN, students))
    diag['num_sessions'] = len(sessions)

    # 1) sessions that cannot fit any single room (for non-lab sessions)
    impossible_sessions = []
    for s in sessions:
        idx, typ, dur, students = s
        if typ != 'P' and students > max_room_cap:
            code = all_courses[idx].get('code', 'UNKNOWN')
            impossible_sessions.append({'code': code, 'type': typ, 'students': students, 'max_room': max_room_cap})
    if impossible_sessions:
        problems.append("One or more (L/T) sessions need more seats than the largest available room.")
        for e in impossible_sessions:
            problems.append(f"  • {e['code']} ({e['type']}) needs {e['students']} seats but largest room holds {e['max_room']}.")
        problems.append("Fix: allow big halls for class scheduling or reduce 'students' or split the course (splitting not supported automatically).")
        return problems, diag

    # 2) for lab sessions (P) ensure total lab capacity across L-rooms can accommodate largest lab session
    largest_lab_need = 0
    for s in sessions:
        idx, typ, dur, students = s
        if typ == 'P' and students > largest_lab_need:
            largest_lab_need = students
    if largest_lab_need > total_lab_cap:
        problems.append("One or more lab sessions have more students than total available lab seating across all 'L' rooms.")
        problems.append(f"  • largest lab session requires {largest_lab_need} seats, but total lab room capacity is {total_lab_cap}.")
        problems.append("Fix: add/allow more lab rooms or increase capacities.")
        return problems, diag

    # 3) compute required units per (year,component group)
    
    core_units_by_group = defaultdict(int)
    # NEW: Track by (year, comp, basket)
    elective_units_by_group = defaultdict(lambda: {'L': 0, 'T': 0, 'P': 0})

    for idx, c in enumerate(all_courses):
        L_count = int(c.get('L', 0) or 0)
        T_count = int(c.get('T', 0) or 0)
        P_count = int(c.get('P', 0) or 0)
        
        L_units = L_count * (LECTURE_MIN // TIME_UNIT_MIN)
        T_units = T_count * (TUTORIAL_MIN // TIME_UNIT_MIN)
        P_units = P_count * (LAB_MIN // TIME_UNIT_MIN)
        
        total_core_course_units = L_units + T_units + P_units
        
        # Extract Basket
        elec_val = str(c.get('elective', '')).strip().upper()
        is_elective = elec_val not in ['', 'FALSE', '0', 'NO', 'NAN']
        basket = '1'
        if is_elective:
            match = re.search(r'\d+', elec_val)
            if match:
                basket = match.group(0)

        group_field = str(c.get('group', c.get('section', 'COMMON')))
        components = [g.strip() for g in group_field.split(';')] if ';' in group_field else [group_field]
        year = int(c.get('year', 1))
        
        for comp in components:
            if is_elective:
                key = (year, comp, basket)
                elective_units_by_group[key]['L'] = max(elective_units_by_group[key]['L'], L_units)
                elective_units_by_group[key]['T'] = max(elective_units_by_group[key]['T'], T_units)
                elective_units_by_group[key]['P'] = max(elective_units_by_group[key]['P'], P_units)
            else:
                core_units_by_group[(year, comp)] += total_core_course_units

    required_units_by_group = defaultdict(int)
    # Combine core units and the maximum required parallel elective units PER BASKET
    for (y, c), units in core_units_by_group.items():
        required_units_by_group[(y, c)] += units
    for (y, c, b), e_dict in elective_units_by_group.items():
        required_units_by_group[(y, c)] += e_dict['L'] + e_dict['T'] + e_dict['P']

    # any group requiring more core-week units than available?
    oversubscribed = []
    for (yr, grp), req in required_units_by_group.items():
        if req > total_core_week_units:
            oversubscribed.append({'year': yr, 'group': grp, 'required_units': req, 'core_units': total_core_week_units})
    if oversubscribed:
        problems.append("One or more groups require more teaching time-units than the core weekly window provides.")
        for o in oversubscribed:
            problems.append(f"  • Year{o['year']} {o['group']} needs {o['required_units']} units but core-week provides {o['core_units']}.")
        problems.append("Fix: move some sessions to electives/evening, reduce counts, or reduce duplicated/combined rows.")

    # 4) professor load sanity check (rough)
    prof_units = defaultdict(int)
    for c in all_courses:
        instrs = _normalize_instructors(c.get('instructor',''))
        if not instrs: instrs = ['TBD']
        units = (int(c.get('L',0) or 0) * (LECTURE_MIN // TIME_UNIT_MIN) +
                 int(c.get('T',0) or 0) * (TUTORIAL_MIN // TIME_UNIT_MIN) +
                 int(c.get('P',0) or 0) * (LAB_MIN // TIME_UNIT_MIN))
        for p in instrs:
            prof_units[p] += units
    heavy_profs = []
    for p, u in prof_units.items():
        minutes = u * TIME_UNIT_MIN
        week_minutes = total_week_units * TIME_UNIT_MIN
        if minutes > week_minutes:
            heavy_profs.append({'prof': p, 'minutes': minutes, 'week_minutes': week_minutes})
    if heavy_profs:
        problems.append("Professor teaching load seems impossible (sum of assigned sessions exceeds total available minutes in the week).")
        for hp in heavy_profs:
            problems.append(f"  • {hp['prof']} assigned {hp['minutes']} minutes > whole-week {hp['week_minutes']} minutes.")
        problems.append("Fix: check instructor field duplicates or reassign.")

    return problems, diag


def generate_timetable_fast(all_courses, rooms):
    """Run preflight diagnostics first. If passes, run the CP-SAT solver."""
    problems, diag = _preflight_checks(all_courses, rooms)
    st.info("Preflight diagnostics: " + ", ".join(f"{k}={v}" for k,v in diag.items()))
    if problems:
        st.error("Preflight checks found issues. Solver not executed.")
        for p in problems:
            st.error(p)
        return []

    model = cp_model.CpModel()
    sessions = decompose_sessions(all_courses)
    
    # --- Linear Time Mapping ---
    units_per_day = (DAY_FULL_END_MIN - DAY_FIRST_MIN) // TIME_UNIT_MIN
    num_units_total = units_per_day * len(DAYS)
    
    lunch_start_unit = (LUNCH_START_MIN - DAY_FIRST_MIN) // TIME_UNIT_MIN
    lunch_end_unit = (LUNCH_START_MIN + LUNCH_DURATION_MIN - DAY_FIRST_MIN) // TIME_UNIT_MIN
    core_end_unit = (CLASSES_END_MIN - DAY_FIRST_MIN) // TIME_UNIT_MIN

    S = len(sessions)
    if S == 0:
        st.warning("No teaching sessions detected.")
        return []

    # Timing vars
    starts = [model.NewIntVar(0, num_units_total - 1, f's_{i}') for i in range(S)]
    base_durations_units = [s['duration_min'] // TIME_UNIT_MIN for s in sessions]
    durations = [d + BREAK_UNITS for d in base_durations_units]
    ends = [model.NewIntVar(0, num_units_total, f'e_{i}') for i in range(S)]
    intervals = [model.NewIntervalVar(starts[i], durations[i], ends[i], f'i_{i}') for i in range(S)]
    day_vars = [model.NewIntVar(0, len(DAYS) - 1, f'd_{i}') for i in range(S)]

    prof_intervals = defaultdict(list)
    group_intervals = defaultdict(list)
    course_activity_groups = defaultdict(list)
    room_optional_intervals = {r_idx: [] for r_idx in range(len(rooms))}
    elective_basket_vars = {}
    
    group_elective_tracked = set()
    session_indices_by_group = defaultdict(list)

    # prepare room capacities and lab flags
    room_caps = []
    room_is_lab = []
    room_names = []
    for r in rooms:
        try:
            rc = int(r.get('capacity', 0) or 0)
        except Exception:
            rc = 0
        room_caps.append(rc)
        name = str(r.get('name','')).strip()
        room_names.append(name)
        room_is_lab.append(True if name.upper().startswith('L') else False)

    is_in_room = {}
    for i in range(S):
        for r_idx in range(len(rooms)):
            is_in_room[(i, r_idx)] = model.NewBoolVar(f"in_s{i}_r{r_idx}")

    # SINGLE LOOP for all constraints!
    for i, s in enumerate(sessions):
        course = all_courses[s['course_idx']]
        elective = s['elective']
        group = str(course.get('group', course.get('section', 'COMMON')))
        
        day_offset = day_vars[i] * units_per_day

        # 1. Basic Day Constraints
        model.Add(starts[i] >= day_offset)
        model.Add(ends[i] <= day_offset + units_per_day)

        # --- NEW: Strict Lunch Separation ---
        is_before_lunch = model.NewBoolVar(f'before_lunch_{i}')
        model.Add(ends[i] <= day_offset + lunch_start_unit).OnlyEnforceIf(is_before_lunch)
        model.Add(starts[i] >= day_offset + lunch_end_unit).OnlyEnforceIf(is_before_lunch.Not())

        # Core window constraint for non-electives
        if not elective:
            model.Add(ends[i] <= day_offset + core_end_unit)

        # professor no-overlap
        instrs = _normalize_instructors(course.get('instructor',''))
        if not instrs:
            instrs = ['TBD']
        for instr in instrs:
            prof_intervals[instr].append(intervals[i])

        # GROUP NO-OVERLAP LOGIC
        comps = [p.strip() for p in str(group).split(';')] if ';' in str(group) else [str(group)]
        for comp in comps:
            year = int(course.get('year',1))
            
            if elective:
                act = s['activity']
                occ = s.get('occurrence', 0)
                basket = s['basket']
                key = f"global_elec_Y{year}_B{basket}_{act}_occ{occ}"
                
                track_key = (year, comp, key)
                if track_key not in group_elective_tracked:
                    group_intervals[(year, comp)].append(intervals[i])
                    group_elective_tracked.add(track_key)
            else:
                group_intervals[(year, comp)].append(intervals[i])
                
            session_indices_by_group[(year, comp)].append(i)

        course_activity_groups[(s['course_idx'], s['activity'])].append(i)

        # room assignment logic
        students = int(course.get('students', 0) or 0)
        activity = s['activity']

        for r_idx, room in enumerate(rooms):
            opt_int = model.NewOptionalIntervalVar(starts[i], durations[i], ends[i], is_in_room[(i, r_idx)], f'opt_s{i}_r{r_idx}')
            room_optional_intervals[r_idx].append(opt_int)

            if activity in ('L', 'T'):
                if room_is_lab[r_idx]:
                    model.Add(is_in_room[(i, r_idx)] == 0)
                if room_caps[r_idx] < students:
                    model.Add(is_in_room[(i, r_idx)] == 0)
            elif activity == 'P':
                if not room_is_lab[r_idx]:
                    model.Add(is_in_room[(i, r_idx)] == 0)
            else:
                if room_caps[r_idx] < students:
                    model.Add(is_in_room[(i, r_idx)] == 0)

        if activity in ('L', 'T'):
            model.Add(sum(is_in_room[(i, r_idx)] for r_idx in range(len(rooms))) == 1)
        elif activity == 'P':
            model.Add(sum(is_in_room[(i, r_idx)] for r_idx in range(len(rooms))) >= 1)
            model.Add(sum(is_in_room[(i, r_idx)] * room_caps[r_idx] for r_idx in range(len(rooms))) >= students)

        # ELECTIVE LOGIC: Force parallel execution
        if elective:
            year = int(course.get('year', 1))
            occ = s.get('occurrence', 0)
            act = s['activity']
            basket = s['basket']
            
            key = f"global_elec_Y{year}_B{basket}_{act}_occ{occ}"
            
            if key not in elective_basket_vars:
                elective_basket_vars[key] = model.NewIntVar(0, num_units_total - 1, key)
            
            model.Add(starts[i] == elective_basket_vars[key])

    # No-overlap constraints
    for iv_list in prof_intervals.values():
        if len(iv_list) > 1:
            model.AddNoOverlap(iv_list)
    for iv_list in group_intervals.values():
        if len(iv_list) > 1:
            model.AddNoOverlap(iv_list)
    for iv_list in room_optional_intervals.values():
        if len(iv_list) > 1:
            model.AddNoOverlap(iv_list)
    for session_indices in course_activity_groups.values():
        if len(session_indices) > 1:
            model.AddAllDifferent([day_vars[i] for i in session_indices])
            
    # CONSTRAINT: Ensure every group has at least one class every day
    for (year, group_name), session_indices in session_indices_by_group.items():
        if not session_indices:
            continue
            
        for day_idx in range(len(DAYS)):
            is_session_on_day = []
            for i in session_indices:
                is_on_day_i = model.NewBoolVar(f'is_Y{year}_{group_name}_S{i}_D{day_idx}')
                model.Add(day_vars[i] == day_idx).OnlyEnforceIf(is_on_day_i)
                model.Add(day_vars[i] != day_idx).OnlyEnforceIf(is_on_day_i.Not())
                is_session_on_day.append(is_on_day_i)
                
            model.Add(sum(is_session_on_day) >= 1).WithName(f'DailyClass_Y{year}_{group_name}_D{day_idx}')

    # --- NEW LOGIC: Stop Lab Room Hogging ---
    total_rooms_used = sum(is_in_room[(i, r_idx)] for i in range(S) for r_idx in range(len(rooms)))
    model.Minimize(total_rooms_used)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 15
    solver.parameters.max_time_in_seconds = 300
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        st.error("Solver finished with no solution. Try relaxing constraints or inspect preflight errors above.")
        return []

    # --- UPDATED: Linear time extraction to fix gap shifting ---
    def unit_to_minute(start_val):
        day_idx = start_val // units_per_day
        unit_index = start_val % units_per_day
        start_min = DAY_FIRST_MIN + unit_index * TIME_UNIT_MIN
        return day_idx, start_min

    assignments = []
    for i, s in enumerate(sessions):
        start_val = solver.Value(starts[i])
        day_idx, start_min = unit_to_minute(start_val)
        course = all_courses[s['course_idx']]
        chosen_rooms = []
        for r_idx in range(len(rooms)):
            val = solver.Value(is_in_room[(i, r_idx)])
            if val == 1:
                chosen_rooms.append(room_names[r_idx])
        room_name = ", ".join(chosen_rooms) if chosen_rooms else "TBD"
        assignments.append({
            'course': course.get('code',''),
            'instructor': course.get('instructor',''),
            'day': DAYS[day_idx],
            'start_min': start_min,
            'end_min': start_min + s['duration_min'],
            'room': room_name,
            'activity': s['activity'],
            'year': int(course.get('year',1)),
            'group': str(course.get('group', course.get('section', 'COMMON'))),
            'elective': s.get('elective', False),
            'basket': s.get('basket', '1')
        })
    return assignments


# -------------------------
# EXAM SCHEDULER (group aware) -- EXCLUDES BIG HALLS (per user request)
# -------------------------
def generate_exam_timetable(all_courses_records, rooms, exam_days, exam_duration_min=180, exclude_rooms_for_exam=None, max_exams_per_slot=MAX_EXAMS_PER_SLOT_DEFAULT):
    if exclude_rooms_for_exam is None:
        exclude_rooms_for_exam = set()
    
    # Use 9am (540 min) and 2pm (840 min)
    slots = [(day, t) for day in exam_days for t in [9*60, 14*60]]
    
    courses = []
    for c in all_courses_records:
        rec = dict(c)
        rec['students'] = int(rec.get('students', 0) or 0)
        rec['year'] = int(rec.get('year', 1) or 1)
        rec['group'] = str(rec.get('group', rec.get('section', 'COMMON')))
        rec['code'] = str(rec.get('code', ''))
        
        # Determine the primary 'Section' for exam grouping/display
        group_str = rec['group'] or ""
        primary_section = group_str.split(';')[0].split('-')[0].strip() if group_str else 'COMMON'
        rec['section_key'] = primary_section.upper()
        
        courses.append(rec)
        
    # Sort by students, code, year, group to prioritize larger, multi-group, and lower-year exams
    courses_sorted = sorted(courses, key=lambda c: (-c['students'], c['code'], c['year'], c['group']))
    assignments = []
    
    # Use (Year, Component Group, Day, Time) for conflict check (allows 2 per day)
    group_slot_used = set() 
    slot_room_used = set()
    # NEW: Track number of exams per slot
    slot_course_count = defaultdict(int)
    
    # prepare rooms for exam (exclude big halls)
    rooms_for_exam = [r for r in rooms if str(r.get('name','')).upper() not in exclude_rooms_for_exam]
    
    for course in courses_sorted:
        group_str = course['group'] or ""
        
        # Determine components that will have conflict
        components = [p.strip() for p in group_str.split(";")] if ";" in group_str else [group_str]
        
        assigned = False
        for day, start_time in slots:
            slot_key = (day, start_time)
            
            # --- NEW CONSTRAINT: Max exams per slot ---
            if slot_course_count[slot_key] >= max_exams_per_slot:
                continue

            # Check for conflict: If any component group has an exam at this specific (Day, Time) slot
            conflict = False
            for comp in components:
                # Use (year, component, day, time) as the unique slot
                if ((course['year'], comp), day, start_time) in group_slot_used:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            # Room assignment check: find a suitable room, even if we don't include it in the output table
            suitable_rooms = sorted([r for r in rooms_for_exam if int(r.get('capacity', 0) or 0) >= course['students']], key=lambda r: int(r.get('capacity', 0)))
            
            # Find an available room for capacity check
            room = next((r for r in suitable_rooms if (day, start_time, r['name']) not in slot_room_used), None)
            
            # If a session requires a room but none are available, skip this slot.
            if course['students'] > 0 and not room:
                continue
                
            # Schedule the exam
            assignments.append({
                'Day': day,
                'Time': minutes_to_time(start_time),
                'Course': course['code'],
                'Year': course['year'],
                'Group': course['group'], # Keep the original group for seating/invigilation
                'Section_Key': course['section_key'], # For display grouping
                'Room': room['name'] if room else 'TBD' # Include room for seating tracking
            })
            
            # Update counters if assigned
            for comp in components:
                group_slot_used.add(((course['year'], comp), day, start_time))
            
            if room:
                 slot_room_used.add((day, start_time, room['name'])) 
            
            slot_course_count[slot_key] += 1 # INCREMENT COUNT
                 
            assigned = True
            break # Move to next course

    df = pd.DataFrame(assignments, columns=['Day', 'Time', 'Course', 'Year', 'Group', 'Section_Key', 'Room'])
    if not df.empty:
        df['Section'] = df['Section_Key'] # Redundant, but kept for old code compatibility
        df = df.drop(columns=['Section_Key'])
    return df

# -------------------------
# TIMETABLE HTML (group-aware)
# -------------------------


def generate_html_timetable(assignments, year, group, course_colors):
    def assignment_applies_to_group(a, year, group):
        if a.get('year') != year:
            return False
        g = str(a.get('group','') or '')
        if g == group:
            return True
        if ';' in g:
            comps = [p.strip() for p in g.split(';') if p.strip()]
            return group in comps
        if '_' in g and group.replace('-', '') in g:
            return True
        return False

    group_assignments = sorted([a for a in assignments if assignment_applies_to_group(a, year, group)], key=lambda x:(DAYS.index(x['day']), x['start_min']))
    header = "".join([f'<th colspan="{60//TIME_UNIT_MIN}">{minutes_to_time(t)}</th>' for t in range(7*60, DAY_FULL_END_MIN, 60)])
    
    html = f"""
    <style>
    .tt-wrapper {{ overflow-x: auto; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid #333; }}
    .tt {{ width: 100%; min-width: 1300px; border-collapse: collapse; font-family: 'Segoe UI', system-ui, sans-serif; color: #e0e0e0; table-layout: fixed; font-size: 13px; }}
    .tt th, .tt td {{ border: 1px solid #333; text-align: center; padding: 0; height: 85px; vertical-align: middle; }}
    .tt thead th {{ background: #1a1a1a; padding: 12px; font-weight: 600; color: #ccc; border-bottom: 2px solid #444; position: sticky; top: 0; z-index: 1; }}
    .day-col {{ background: #161616; width: 70px; font-weight: 700; color: #ffaa00; border-right: 2px solid #444; text-transform: uppercase; letter-spacing: 1px; }}
    
    .slot-container {{ padding: 4px; height: 100%; width: 100%; box-sizing: border-box; }}
    .slot {{ border-radius: 6px; padding: 6px; display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; box-sizing: border-box; box-shadow: 0 2px 4px rgba(0,0,0,0.4); transition: transform 0.15s ease, box-shadow 0.15s ease; overflow: hidden; }}
    .slot:hover {{ transform: scale(1.03); z-index: 10; position: relative; box-shadow: 0 6px 12px rgba(0,0,0,0.6); }}
    
    .slot b {{ font-size: 1em; margin-bottom: 3px; line-height: 1.1; text-align: center; }}
    .slot small {{ font-size: 0.8em; opacity: 0.9; line-height: 1.3; display: block; text-align: center; }}
    
    .lunch {{ background: repeating-linear-gradient(45deg, #1c1c1c, #1c1c1c 10px, #141414 10px, #141414 20px); color: #666; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; }}
    .minor {{ background: #161616; color: #444; font-style: italic; font-size: 0.9em; }}
    </style>
    <div class="tt-wrapper">
    <table class="tt"><thead><tr><th class="day-col">Day</th>{header}</tr></thead><tbody>
    """
    
    elective_legend_data = set()
    assignments_by_day = {day:[a for a in group_assignments if a['day']==day] for day in DAYS}
    
    for day in DAYS:
        html += f'<tr><td class="day-col">{day}</td>'
        current_minute = 7*60
        class_idx = 0
        day_schedule = assignments_by_day[day]
        
        if current_minute < DAY_FIRST_MIN:
            span = (DAY_FIRST_MIN - current_minute)//TIME_UNIT_MIN
            html += f'<td colspan="{span}" class="minor">Minor / Major Slot</td>'
            current_minute = DAY_FIRST_MIN
            
        while current_minute < DAY_FULL_END_MIN:
            if LUNCH_START_MIN <= current_minute < LUNCH_START_MIN + LUNCH_DURATION_MIN:
                span = LUNCH_DURATION_MIN // TIME_UNIT_MIN
                html += f'<td colspan="{span}" class="lunch">Lunch</td>'
                current_minute += LUNCH_DURATION_MIN
            else:
                concurrent_classes = [c for c in day_schedule[class_idx:] if c['start_min'] == current_minute]
                
                if concurrent_classes:
                    duration = concurrent_classes[0]['end_min'] - concurrent_classes[0]['start_min']
                    span = max(1, duration // TIME_UNIT_MIN)
                    is_elective_block = concurrent_classes[0].get('elective', False)
                    
                    # IF STANDARD CORE SUBJECT
                    if not is_elective_block:
                        next_class = concurrent_classes[0]
                        display_group = next_class.get('group','')
                        if ';' in display_group:
                            display_group = " & ".join([p.strip() for p in display_group.split(';') if p.strip()])
                        
                        bg, border = course_colors.get(next_class['course'], ("#2a2a2a","#cccccc"))
                        
                        # UPDATED TEXT COLORS FOR DARK BACKGROUNDS
                        info = (f"<b style='color: #ffffff;'>{next_class['course']} ({next_class['activity']})</b>"
                                f"<small style='color: #bbbbbb;'>{minutes_to_time(next_class['start_min'])}–{minutes_to_time(next_class['end_min'])}</small>"
                                f"<small style='color: #dddddd;'><b>{next_class['room']}</b> | {next_class['instructor']}</small>"
                                f"<small style='color: #999999;'>{display_group}</small>")
                        
                        html += f'<td colspan="{span}"><div class="slot-container"><div class="slot" style="border-left: 5px solid {border}; background: {bg}; border-top: 1px solid #333; border-right: 1px solid #333; border-bottom: 1px solid #333;">{info}</div></div></td>'
                    
                    # IF ELECTIVE BLOCK
                    else:
                        basket_id = concurrent_classes[0].get('basket', '1')
                        for c in concurrent_classes:
                            elective_legend_data.add((basket_id, c['course'], c['instructor'], c['day'], c['start_min'], c['end_min'], c['room']))
                        
                        info = (f"<b style='color: #ffaa00; letter-spacing: 0.5px;'>BASKET {basket_id} ELECTIVES ({concurrent_classes[0]['activity']})</b>"
                                f"<small style='color: #bbb;'>{minutes_to_time(concurrent_classes[0]['start_min'])}–{minutes_to_time(concurrent_classes[0]['end_min'])}</small>"
                                f"<small style='color: #888; margin-top: 4px;'>(See Allocation Table Below)</small>")
                        
                        html += f'<td colspan="{span}"><div class="slot-container"><div class="slot" style="border-left: 5px solid #ffaa00; background: #1a1a1a; border: 1px solid #333; border-left: 5px solid #ffaa00;">{info}</div></div></td>'
                    
                    current_minute += duration
                    class_idx += len(concurrent_classes)
                else:
                    remaining_classes = day_schedule[class_idx:]
                    if current_minute >= CLASSES_END_MIN and not remaining_classes:
                        span = (DAY_FULL_END_MIN - current_minute)//TIME_UNIT_MIN
                        html += f'<td colspan="{span}" class="minor">Minor / Major Slot</td>'
                        current_minute = DAY_FULL_END_MIN
                    else:
                        html += '<td></td>'
                        current_minute += TIME_UNIT_MIN
        html += '</tr>'
    html += "</tbody></table></div>"
    
    # --- PROFESSIONAL TABLE LEGEND ---
    if elective_legend_data:
        baskets = {}
        instructors = {}
        for basket, course, instructor, day, start_min, end_min, room in elective_legend_data:
            if basket not in baskets:
                baskets[basket] = {}
            if course not in baskets[basket]:
                baskets[basket][course] = []
            
            baskets[basket][course].append({'day': day, 'start': start_min, 'end': end_min, 'room': room})
            instructors[course] = instructor

        html += "<div style='margin-top: 25px; margin-bottom: 30px; padding: 20px; background-color: #1a1a1a; border-radius: 8px; border: 1px solid #333; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>"
        html += "<h3 style='margin: 0 0 20px 0; color: #fff; font-family: Segoe UI, sans-serif; font-size: 1.2em; border-bottom: 1px solid #333; padding-bottom: 10px;'>Elective Room Allocations</h3>"
        
        day_order = {d: i for i, d in enumerate(DAYS)}
        
        for basket in sorted(baskets.keys()):
            html += f"<h4 style='color: #ffaa00; font-family: Segoe UI, sans-serif; margin: 0 0 10px 0; font-size: 1em; text-transform: uppercase;'>Basket {basket}</h4>"
            html += "<div style='overflow-x: auto; margin-bottom: 25px;'>"
            html += "<table style='width: 100%; border-collapse: collapse; font-family: Segoe UI, sans-serif; font-size: 13px; color: #eee; text-align: left; border: 1px solid #333;'>"
            html += "<thead><tr style='background-color: #222; border-bottom: 2px solid #444;'>"
            html += "<th style='padding: 10px; border-right: 1px solid #333; width: 35%;'>Course</th>"
            html += "<th style='padding: 10px; border-right: 1px solid #333; width: 25%;'>Instructor</th>"
            html += "<th style='padding: 10px; width: 40%;'>Schedule & Rooms</th>"
            html += "</tr></thead><tbody>"
            
            for course in sorted(baskets[basket].keys()):
                inst = instructors[course]
                sessions = baskets[basket][course]
                sessions.sort(key=lambda x: (day_order.get(x['day'], 99), x['start']))
                
                schedule_strs = []
                for s in sessions:
                    t_start = minutes_to_time(s['start'])
                    t_end = minutes_to_time(s['end'])
                    schedule_strs.append(f"<span style='display: inline-block; background: #2a2a2a; padding: 4px 8px; border-radius: 4px; margin: 3px 3px 3px 0; border: 1px solid #444; white-space: nowrap;'>{s['day']} {t_start}-{t_end} &rarr; <b style='color:#fff;'>{s['room']}</b></span>")
                    
                sched_html = "".join(schedule_strs)
                
                html += f"<tr style='border-bottom: 1px solid #333; background-color: #1c1c1c; transition: background-color 0.15s ease;' onmouseover='this.style.backgroundColor=\"#262626\"' onmouseout='this.style.backgroundColor=\"#1c1c1c\"'>"
                html += f"<td style='padding: 12px 10px; border-right: 1px solid #333; font-weight: 600;'>{course}</td>"
                html += f"<td style='padding: 12px 10px; border-right: 1px solid #333; color: #ccc;'>{inst}</td>"
                html += f"<td style='padding: 8px 10px;'>{sched_html}</td>"
                html += "</tr>"
                
            html += "</tbody></table></div>"
            
        html += "</div>"

    return html
# -------------------------
# EXAM TIMETABLE DISPLAY FUNCTIONS
# -------------------------
def generate_simple_exam_timetable_df(exam_df):
    if exam_df.empty:
        return pd.DataFrame(columns=['Day', 'Time', 'Course', 'Group'])
    
    # Extract the main section (e.g., 'CSE' from 'CSE-A' or 'CSE-A;CSE-B')
    def get_main_section(group_str):
        if not group_str:
            return 'COMMON'
        return group_str.split(';')[0].split('-')[0].strip().upper()

    exam_df['Main_Section'] = exam_df['Group'].apply(get_main_section)
    
    # Drop Room column if it exists and keep only relevant columns for display
    display_df = exam_df.drop(columns=['Room', 'Section'], errors='ignore')
    display_df = display_df.rename(columns={'Course': 'Subject', 'Time': 'Slot'})
    
    # Sort for consistent display order
    display_df['DayNum'] = (
        display_df['Day']
        .astype(str)
        .str.extract(r'(\d+)')
        .fillna(-1)
        .astype(int)
    )
    
    display_df = display_df.sort_values(['Year', 'Main_Section', 'DayNum', 'Slot'], ignore_index=True)
    
    # Select final columns for output
    final_cols = ['Year', 'Main_Section', 'Day', 'Slot', 'Subject', 'Group']
    return display_df[final_cols]


# Function to generate a single CSV from all group timetables

def generate_all_timetables_csv(assignments, all_courses_df):
    if not assignments:
        return pd.DataFrame().to_csv(index=False)

    assignments_df = pd.DataFrame(assignments)

    # --- UPDATED: Get all unique (year, group) combinations, correctly unpacking combined strings ---
    unique_combinations = set()
    for yr, grp_str in all_courses_df[['year', 'group']].drop_duplicates().itertuples(index=False, name=None):
        if pd.isna(grp_str):
            continue
        for comp in str(grp_str).split(';'):
            comp = comp.strip()
            if comp:
                unique_combinations.add((int(yr), comp))
                
    combos = sorted(list(unique_combinations))
    
    all_group_timetables = []
    
    # Helper to check if an assignment applies to a specific single group
    def assignment_applies_to_group(a, year, group):
# ... (leave the rest of the function unchanged)
        if a.get('year') != year:
            return False
        g = str(a.get('group','') or '')
        # Check if it's the specific group OR if the specific group is a component
        if g == group:
            return True
        if ';' in g:
            comps = [p.strip() for p in g.split(';') if p.strip()]
            return group in comps
        return False
    
    for yr, grp in combos:
        # Filter assignments for the current (year, group)
        group_assignments = assignments_df[
            assignments_df.apply(lambda a: assignment_applies_to_group(a, yr, grp), axis=1)
        ].copy()

        if not group_assignments.empty:
            group_assignments['Target_Year'] = yr
            group_assignments['Target_Group'] = grp
            all_group_timetables.append(group_assignments)
            
    if not all_group_timetables:
        return pd.DataFrame().to_csv(index=False)
        
    final_df = pd.concat(all_group_timetables, ignore_index=True)
    
    # Map start_min to time format
    final_df['Start_Time'] = final_df['start_min'].apply(minutes_to_time)
    final_df['End_Time'] = final_df['end_min'].apply(minutes_to_time)

    # Clean up and select final columns
    final_cols = [
        'Target_Year', 'Target_Group', 'day', 'Start_Time', 'End_Time', 
        'course', 'activity', 'room', 'group', 'instructor'
    ]
    
    final_df = final_df[final_cols].rename(columns={
        'day': 'Day', 
        'course': 'Course', 
        'activity': 'Activity', 
        'room': 'Room(s)', 
        'group': 'Actual_Assigned_Group(s)',
        'instructor': 'Instructor'
    })
    
    # Sort for better readability
    final_df = final_df.sort_values(by=['Target_Year', 'Target_Group', 'Day', 'Start_Time']).reset_index(drop=True)
    
    return final_df.to_csv(index=False).encode('utf-8')


# -------------------------
# SEATING SUMMARY (group-aware)
# -------------------------
def seating_summary_two_groups_per_room(exam_df, rooms, all_courses_df, exclude_rooms=None, seed=42):
    if exclude_rooms is None:
        exclude_rooms = set()
    exclude_upper = {s.upper() for s in exclude_rooms}
    usable_rooms = []
    for r in rooms:
        name = str(r['name']).strip()
        if name.upper() in exclude_upper:
            continue
        if name.upper().startswith('L'):
            continue
        # Use seating capacity as half of original capacity
        eff = int((int(r.get('capacity', 0)) // 2))
        if eff <= 0:
            continue
        usable_rooms.append({'name': name, 'orig': int(r.get('capacity', 0)), 'eff': eff})
    usable_rooms = sorted(usable_rooms, key=lambda x: x['eff'], reverse=True)

    days = sorted(exam_df['Day'].unique(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)) if not exam_df.empty else []
    
    # Find all unique time slots across all days
    all_time_slots = sorted(exam_df['Time'].unique())
    
    # Seating summary will be generated for Day * Time * Room
    summary_rows = []
    
    for day_idx, day in enumerate(days):
        for time_slot in all_time_slots:
            # Get all exams on this specific Day and Time
            day_time_exams = exam_df[(exam_df['Day'] == day) & (exam_df['Time'] == time_slot)]
            
            # 1. Calculate remaining students per (year, group) for this slot
            rem = defaultdict(int)
            for _, row in day_time_exams.iterrows():
                code = row['Course']
                year = int(row['Year'])
                group = str(row.get('Group', row.get('Section', 'COMMON')))
                
                # Find student count from the original course data
                matched = all_courses_df[
                    (all_courses_df['code'] == code) &
                    (all_courses_df['year'].astype(int) == year) &
                    (all_courses_df.get('group', all_courses_df.get('section', 'COMMON')).astype(str) == group)
                ]
                if not matched.empty:
                    students = int(matched.iloc[0].get('students', 0))
                else:
                    mc = all_courses_df[all_courses_df['code'] == code]
                    students = int(mc['students'].max()) if not mc.empty else 0
                    
                rem[(year, group)] += students

            counts = dict(rem)
            groups_to_assign = [k for k, v in counts.items() if v > 0]
            
            # Shuffle groups per slot for randomization (use a unique seed per slot)
            rnd = random.Random(seed + day_idx + hash(time_slot) % 1000)
            rnd.shuffle(groups_to_assign)
            counts_sorted = {k: counts[k] for k in groups_to_assign}

            branch_serial_map = defaultdict(int) # Reset rolls for each slot

            # 2. Assign groups to rooms
            for room in usable_rooms:
                room_name = room['name']
                eff = room['eff']
                
                # If no students remain, fill with blank rows
                if not counts_sorted:
                    summary_rows.append({'Day': day, 'Time': time_slot, 'Room': room_name, 'Group A': '', 'Rolls A': '', 'Group B': '', 'Rolls B': '', 'RoomEff': eff})
                    continue
                
                half1 = eff // 2 + (eff % 2)
                half2 = eff - half1
                
                # Sort remaining groups by size for assignment priority
                sorted_groups = sorted(counts_sorted.items(), key=lambda kv: (-kv[1], kv[0][0], str(kv[0][1])))

                # Group A: Largest remaining group
                g1_key, g1_avail = sorted_groups[0]
                assign1 = min(half1, g1_avail)

                # Group B: Second largest remaining group, or fill Group A if capacity allows
                g2_key = None
                assign2 = 0
                
                # Check for a different group first (Group B)
                for k, v in sorted_groups[1:]:
                    if k != g1_key and v > 0:
                        # Ensure we don't seat a group in a room that also hosts a common exam they are taking.
                        # This simple model assumes the exam_df only contains one exam per group per slot.
                        g2_key = k
                        assign2 = min(half2, v)
                        break
                        
                # If no second group, and Group A still has students, use Group A for the second half
                if g2_key is None and half2 > 0:
                    if g1_avail - assign1 > 0:
                        assign2 = min(half2, g1_avail - assign1)
                        g2_key = g1_key # Group B is the overflow of Group A

                gA_str, rA_str, gB_str, rB_str = "", "", "", ""

                if assign1 > 0:
                    yA, grpA = g1_key
                    gA_str = f"Y{int(yA)} {grpA}"
                    rA_str = allocate_rolls(yA, grpA, assign1, branch_serial_map)
                    counts_sorted[g1_key] -= assign1
                    if counts_sorted[g1_key] <= 0:
                        counts_sorted.pop(g1_key, None)

                if assign2 > 0 and g2_key is not None:
                    yB, grpB = g2_key
                    gB_str = f"Y{int(yB)} {grpB}"
                    rB_str = allocate_rolls(yB, grpB, assign2, branch_serial_map)
                    
                    if g2_key in counts_sorted:
                        counts_sorted[g2_key] -= assign2
                        if counts_sorted[g2_key] <= 0:
                            counts_sorted.pop(g2_key, None)
                            
                # Only add a row if there was a group assigned OR the room is one of the smallest available rooms (to avoid filling all rooms with blanks)
                if assign1 > 0 or assign2 > 0 or room_name in [r['name'] for r in usable_rooms[:5]]:
                    summary_rows.append({'Day': day, 'Time': time_slot, 'Room': room_name, 'Group A': gA_str, 'Rolls A': rA_str, 'Group B': gB_str, 'Rolls B': rB_str, 'RoomEff': eff})

    summary_df = pd.DataFrame(summary_rows, columns=['Day', 'Time', 'Room', 'Group A', 'Rolls A', 'Group B', 'Rolls B', 'RoomEff'])
    
    # Final cleanup: drop rows with no assignments unless it is for one of the main time slots and rooms
    if not summary_df.empty:
        summary_df = summary_df.drop_duplicates(subset=['Day', 'Time', 'Room'], keep='first')
        summary_df['Total Assigned'] = summary_df[['Group A', 'Group B']].apply(lambda x: 1 if x['Group A'] else 0 + 1 if x['Group B'] else 0, axis=1)
        summary_df = summary_df.sort_values(by=['Day', 'Time', 'Total Assigned'], ascending=[True, True, False])
        summary_df = summary_df.drop(columns=['Total Assigned'])

    return summary_df

def allocate_rolls(year, group, count, branch_serial_map):
    """Generates roll numbers for a specific year and group within a single slot."""
    if count <= 0:
        return ""
    start = generate_roll_number_v2(year, group, branch_serial_map)
    last = start
    for _ in range(count - 1):
        last = generate_roll_number_v2(year, group, branch_serial_map)
    return f"{start} - {last}"

# -------------------------
# INVIGILATION ROSTER (from summary)
# -------------------------

def generate_invigilation_roster_from_summary(summary_df, exam_df, all_courses_df, rooms, invig_per_room=2):
    instr_pool = []
    # Loop over all unique instructor fields in the course data
    for instr_field in all_courses_df['instructor'].dropna().unique():
        # Normalize the field (e.g., split by ';')
        for i in _normalize_instructors(instr_field):
            # --- NEW EXCLUSION LOGIC ---
            if i.upper().strip() == 'TBD' or i.upper().strip() == 'T.B.D.':
                continue
            # --- END NEW EXCLUSION LOGIC ---
            
            if i not in instr_pool:
                instr_pool.append(i)
                
    if not instr_pool:
        # If no actual instructors are left after excluding TBD, return empty.
        return pd.DataFrame()
        
    invig_idx = 0
    invig_by_slot = defaultdict(set)
    invig_rows = []
    
    # 1. FILTER: Only consider rooms where a group (A or B) was actually assigned seats.
    roster_eligible_rooms = summary_df[
        (summary_df['Group A'].astype(bool)) | (summary_df['Group B'].astype(bool))
    ].sort_values(by=['Day', 'Time', 'Room'])

    
    for _, row in roster_eligible_rooms.iterrows():
        day = row['Day']
        time_display = row['Time']
        room = row['Room']
        
        # 1. Identify forbidden instructors (those teaching the exams in the room)
        forbidden = set()
        
        # Determine the groups in the room (Group A & B can be the same)
        groups_in_room = []
        if row['Group A']:
            # Example: 'Y1 CSE-A' -> 'CSE-A'
            groups_in_room.append(row['Group A'].split(' ', 1)[1].strip())
        if row['Group B'] and row['Group B'] != row['Group A']:
            groups_in_room.append(row['Group B'].split(' ', 1)[1].strip())

        # Find all courses scheduled for the groups in this room, at this day/time
        relevant_exams = exam_df[(exam_df['Day'] == day) & (exam_df['Time'] == time_display)]

        for _, er in relevant_exams.iterrows():
            code = er['Course']
            year = int(er['Year'])
            group = str(er.get('Group', er.get('Section', 'COMMON')))
            
            # Check if this exam (by its assigned group) applies to any group seated in this room.
            is_relevant_exam = any(g in group for g in groups_in_room)
            
            if is_relevant_exam:
                matched = all_courses_df[
                    (all_courses_df['code'] == code) &
                    (all_courses_df['year'].astype(int) == year) &
                    (all_courses_df.get('group', all_courses_df.get('section', 'COMMON')).astype(str) == group)
                ]
                if not matched.empty:
                    instr_field = matched.iloc[0].get('instructor', '')
                    for i in _normalize_instructors(instr_field):
                        # Ensure TBD itself is not added to the forbidden set, although it wouldn't matter later.
                        if i.upper().strip() != 'TBD' and i.upper().strip() != 'T.B.D.':
                            forbidden.add(i)

        # 2. Assign invigilators
        chosen = []
        tries = 0
        while len(chosen) < invig_per_room and tries < len(instr_pool) + 50:
            if len(instr_pool) == 0:
                break
            
            cand = instr_pool[invig_idx % len(instr_pool)]
            invig_idx += 1
            tries += 1
            
            # Conflict check: forbidden (teaching the exam), used in the same slot, or already chosen for this room
            if cand in forbidden or cand in invig_by_slot[(day, time_display)] or cand in chosen:
                continue
                
            chosen.append(cand)
            invig_by_slot[(day, time_display)].add(cand) # Mark as used for the slot
            
        invig_rows.append({'Day': day, 'Time': time_display, 'Room': room, 'Invigilators': "; ".join(chosen) if chosen else "TBD (No non-conflicting instructor found)"})
        
    invig_df = pd.DataFrame(invig_rows, columns=['Day', 'Time', 'Room', 'Invigilators'])
    return invig_df

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Timetable + Groups", layout="wide")
st.title("📅 University Timetable + Groups (CSE-A / CSE-B / ECE / DSAI)")

st.sidebar.header("Upload CSVs (optional)")
courses_file = st.sidebar.file_uploader("Courses CSV (code,instructor,year,section,students,L,T,P,elective). Optional: 'group' column", type=['csv'])
rooms_file = st.sidebar.file_uploader("Rooms CSV (name,capacity)", type=['csv'])



# -------------------------
# UI CONFIGURATION & LOGIC
# -------------------------
st.sidebar.header("Exam settings")
exam_days_count = st.sidebar.slider("Number of Exam Days", 5, 20, 10)
exam_duration = st.sidebar.slider("Exam Duration (min)", 60, 180, 180)
max_exams_per_slot_ui = st.sidebar.slider("Max Exams per Time Slot", 1, 10, MAX_EXAMS_PER_SLOT_DEFAULT)

# Sample fallback data (updated with 'term' column)
sample_course_csv = """code,instructor,year,section,students,L,T,P,elective,group,term
Statistics,Dr. Ramesh Athe,1,CSE,200,2,0,0,FALSE,CSE-A;CSE-B,pre
Probability,Dr. Chinmayananda,1,ECE,150,2,0,0,FALSE,ECE;DSAI,post
CS101,Dr. Anjali,1,CSE,100,2,1,1,FALSE,CSE-A,full
"""
sample_rooms_csv = """name,capacity
C101,96
C102,96
"""

# Load CSVs
try:
    if courses_file:
        all_courses_df = pd.read_csv(courses_file)
    else:
        all_courses_df = pd.read_csv(pd.io.common.StringIO(sample_course_csv))
except Exception as e:
    st.error(f"Error reading courses file: {e}")
    all_courses_df = pd.DataFrame()

try:
    if rooms_file:
        rooms = pd.read_csv(rooms_file).to_dict(orient='records')
    else:
        rooms = pd.read_csv(pd.io.common.StringIO(sample_rooms_csv)).to_dict(orient='records')
except Exception as e:
    st.error(f"Error reading rooms file: {e}")
    rooms = []

# --- DATA NORMALIZATION ---
if not all_courses_df.empty:
    all_courses_df = normalize_combined_groups(all_courses_df, show_log=True)
    all_courses_df = _finalize_group_rows(all_courses_df)

    # Ensure required columns exist
    required_cols = ['group', 'section', 'year', 'students', 'L', 'T', 'P', 'elective', 'code', 'instructor', 'term']
    for col in required_cols:
        if col not in all_courses_df.columns:
            # Default 'term' to 'full' if missing
            if col == 'term':
                all_courses_df[col] = 'full'
            else:
                all_courses_df[col] = "" if col in ['group', 'section', 'code', 'instructor'] else 0

    # Clean up the term column
    all_courses_df['term'] = all_courses_df['term'].astype(str).str.lower().str.strip().replace({'nan': 'full', '': 'full'})

    # Create group from section if missing
    all_courses_df['group'] = all_courses_df.apply(
        lambda r: (str(r['group']).strip() if str(r['group']).strip() not in ['', 'nan'] else str(r['section']).strip()),
        axis=1
    )
    
    # Safe numeric conversions
    try:
        all_courses_df['year'] = all_courses_df['year'].astype(int)
    except Exception:
        all_courses_df['year'] = pd.to_numeric(all_courses_df['year'], errors='coerce').fillna(1).astype(int)
    all_courses_df['students'] = all_courses_df['students'].fillna(0).astype(int)

st.markdown("---")

# -------------------------
# GENERATION LOGIC
# -------------------------
if st.button("🚀 Generate Timetables & Exams"):
    # Clear previous state
    keys_to_clear = ['assignments_pre', 'assignments_post', 'exam_df', 'all_courses_df', 'rooms', 'generated']
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    
    # 1. GENERATE PRE-MIDSEM TIMETABLE
    with st.spinner("Generating Pre-Midsem Timetable..."):
        # Filter: Keep 'pre' AND 'full' courses
        df_pre = all_courses_df[all_courses_df['term'].isin(['pre', 'full'])].copy()
        courses_pre = df_pre.to_dict(orient='records')
        try:
            assignments_pre = generate_timetable_fast(courses_pre, rooms)
        except Exception as e:
            assignments_pre = []
            st.error(f"Error generating Pre-Midsem: {e}")

    # 2. GENERATE POST-MIDSEM TIMETABLE
    with st.spinner("Generating Post-Midsem Timetable..."):
        # Filter: Keep 'post' AND 'full' courses
        df_post = all_courses_df[all_courses_df['term'].isin(['post', 'full'])].copy()
        courses_post = df_post.to_dict(orient='records')
        try:
            assignments_post = generate_timetable_fast(courses_post, rooms)
        except Exception as e:
            assignments_post = []
            st.error(f"Error generating Post-Midsem: {e}")

    # 3. GENERATE EXAMS (Using ALL courses to catch everything)
    with st.spinner("Generating Exam Schedule..."):
        try:
            all_courses_list = all_courses_df.to_dict(orient='records')
            exam_days = [f"Day {i+1}" for i in range(exam_days_count)]
            exam_df = generate_exam_timetable(
                all_courses_list, rooms, exam_days, 
                exam_duration_min=exam_duration, 
                exclude_rooms_for_exam=BIG_HALLS_TO_EXCLUDE_FOR_EXAMS,
                max_exams_per_slot=max_exams_per_slot_ui
            )
        except Exception as e:
            exam_df = pd.DataFrame()
            st.error(f"Error generating exams: {e}")

    # Store results
    st.session_state['assignments_pre'] = assignments_pre
    st.session_state['assignments_post'] = assignments_post
    st.session_state['exam_df'] = exam_df
    st.session_state['all_courses_df'] = all_courses_df
    st.session_state['rooms'] = rooms
    st.session_state['generated'] = True
    st.success("Generation Complete! Use the switch below to toggle terms.")

# -------------------------
# VIEW LOGIC
# -------------------------
generated = st.session_state.get('generated', False)
assignments_pre = st.session_state.get('assignments_pre', [])
assignments_post = st.session_state.get('assignments_post', [])
exam_df = st.session_state.get('exam_df', pd.DataFrame())
all_courses_df = st.session_state.get('all_courses_df', all_courses_df)

# TABS
tab1, tab2, tab3 = st.tabs(["📚 Class Timetable", "📝 Exam Timetable", "🪑 Seating & Invigilation"])

with tab1:
    if not generated:
        st.info("Click '🚀 Generate' to create schedules.")
    else:
        # --- NEW TOGGLE SWITCH ---
        st.markdown("### 📅 Select Semester Phase")
        term_mode = st.radio("Show Timetable For:", ["Pre-Midsem", "Post-Midsem"], horizontal=True)
        
        # Select the correct assignment set based on toggle
        active_assignments = assignments_pre if term_mode == "Pre-Midsem" else assignments_post
        
        if not active_assignments:
            st.warning(f"No valid schedule found for {term_mode}. Check constraints.")
        else:
            st.success(f"Showing {term_mode} Schedule")
            
            # Recalculate colors for consistency
            course_colors = {c['code']: generate_color_from_string(c['code']) for c in all_courses_df.to_dict(orient='records')}
            
            # Get unique combos
            unique_combinations = set()
            for yr, grp_str in all_courses_df[['year', 'group']].drop_duplicates().itertuples(index=False, name=None):
                if pd.isna(grp_str): continue
                for comp in str(grp_str).split(';'):
                    if comp.strip(): unique_combinations.add((int(yr), comp.strip()))
            combos = sorted(list(unique_combinations))
            
            # Render HTML
            for yr, grp in combos:
                st.subheader(f"Year {yr} — {grp} ({term_mode})")
                try:
                    html = generate_html_timetable(active_assignments, yr, grp, course_colors)
                    st.markdown(html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Render error: {e}")
            
            # Download CSV for ACTIVE term
            csv_data = generate_all_timetables_csv(active_assignments, all_courses_df)
            st.download_button(
                f"📥 Download {term_mode} Timetable CSV", 
                csv_data, 
                file_name=f"timetable_{term_mode.lower().replace('-','_')}.csv", 
                mime='text/csv'
            )

with tab2:
    if exam_df.empty:
        st.info("Generate exam schedule first.")
    else:
        # (Existing Exam Tab Code - Copy from previous version or keep as is)
        st.success("Exam schedule available.")
        relevant_exams = exam_df.copy()
        relevant_exams['DayNum'] = relevant_exams['Day'].astype(str).str.extract(r'(\d+)').fillna(-1).astype(int)
        day_time_slots = relevant_exams[['Day', 'Time', 'DayNum']].drop_duplicates().sort_values(['DayNum', 'Time'])
        
        st.subheader("🗓️ Administrative Exam Schedule")
        for _, row in day_time_slots.iterrows():
            day = row['Day']
            time = row['Time']
            slot_exams = relevant_exams[(relevant_exams['Day'] == day) & (relevant_exams['Time'] == time)].copy()
            slot_exams = slot_exams.sort_values(by=['Year'], ascending=True)
            
            st.markdown(f"**{day} — {time}**")
            display_df = slot_exams[['Course', 'Year', 'Group']].rename(columns={'Course': 'Subject', 'Group': 'Applies To'})
            st.dataframe(display_df, use_container_width=True)

        st.download_button(
            "📥 Download Exam Schedule CSV",
            exam_df.drop(columns=['DayNum'], errors='ignore').to_csv(index=False).encode('utf-8'),
            file_name="exam_schedule.csv", mime='text/csv'
        )

with tab3:
    # (Existing Seating Tab Code - Copy from previous version or keep as is)
    st.header("🪑 Seating & Invigilation")
    if not generated:
        st.info("Click '🚀 Generate' first.")
    else:
        # Defaults
        excludes = DEFAULT_EXCLUDE_ROOMS
        invig_per_room = st.slider("Invigilators per room", 1, 4, 2, key='invig_slider')
        
        if st.button("Generate Seating Plan"):
            with st.spinner("Calculating..."):
                summary_df = seating_summary_two_groups_per_room(exam_df, rooms, all_courses_df, exclude_rooms=excludes)
                invig_df = generate_invigilation_roster_from_summary(summary_df, exam_df, all_courses_df, rooms, invig_per_room=invig_per_room)
                
                if not summary_df.empty:
                    st.dataframe(summary_df, use_container_width=True)
                    st.download_button("Download Seating CSV", summary_df.to_csv(index=False).encode('utf-8'), "seating.csv")
                
                if not invig_df.empty:
                    st.subheader("Invigilation Roster")
                    st.dataframe(invig_df, use_container_width=True)
                    st.download_button("Download Invigilation CSV", invig_df.to_csv(index=False).encode('utf-8'), "invigilation.csv")
