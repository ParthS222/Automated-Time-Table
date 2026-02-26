# Automated-Time-Table

# Automated University Timetable and Exam Scheduler

An automated, constraint-based scheduling system built with Python and Streamlit. This project models the highly complex university timetabling problem and solves it using the Google OR-Tools CP-SAT Solver, generating conflict-free class schedules, exam allocations, and seating arrangements.

## Project Overview

Scheduling university courses and exams involves managing overlapping resources, strict time constraints, and multi-component student groups. This system abstracts these requirements into a Constraint Satisfaction Problem (CSP) and computes optimal or feasible assignments. It features a diagnostic engine to identify specific constraint bottlenecks when a schedule is mathematically infeasible, alongside an interactive web interface for data input and visualization.

## Key Features

* **Algorithmic Class Scheduling:** Utilizes Constraint Programming (CP-SAT) to allocate lectures, tutorials, and labs to available rooms and time slots without overlap.
* **Smart Exam Roster:** Generates exam schedules and automatically handles seating arrangements across multiple rooms, ensuring groups taking the same exam are seated efficiently.
* **Automated Diagnostics Engine:** If a schedule is mathematically impossible, the solver runs isolated tests to pinpoint exactly which constraint (e.g., room capacity, professor overlap) caused the failure.
* **Interactive Interface:** Built with Streamlit for intuitive CSV uploads, parameter adjustment, and real-time visual timetable generation.

## Constraint Modeling

The CP-SAT model strictly enforces the following hard constraints:

1. **No Overlaps:** A professor or student group cannot be assigned to multiple concurrent sessions.
2. **Room Capacity:** Classes cannot be assigned to rooms with a capacity smaller than the registered student count.
3. **Lab Restrictions:** Practical sessions (P) are strictly assigned to designated lab rooms; Lectures (L) and Tutorials (T) cannot use lab rooms.
4. **Lunch and Core Hours:** Enforces strict separation for lunch breaks and adherence to core teaching windows.
5. **Parallel Electives:** Elective courses within the same designated group are forced to run at the exact same time across different rooms.
6. **Room Optimization:** Minimizes the number of unique lab rooms assigned to a single session to prevent unnecessary resource allocation.

## Installation and Setup

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system. 

### 1. Clone the Repository

```bash
git clone [https://github.com/ParthS222/Automated-Time-Table.git](https://github.com/ParthS222/Automated-Time-Table.git)
cd Automated-Time-Table
```

### 2. Install Dependencies

It is recommended to use a virtual environment. Install the required packages using:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Launch the Streamlit application from your terminal:

```bash
streamlit run Timetable.py
```

