# More Than a Hack 2027 — Participation & Repository Guide

> **Event:** More Than a Hack 2027, at Talent Arena  
> **Theme:** Where data meets the future of sports  
> **Official event page:** [landing.mobileworldcapital.com/more-than-a-hack-2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)

This document brings together the published event information and a practical preparation checklist. Sections labeled **Official information** reflect the event page. Sections labeled **Preparation recommendation** are suggestions for our team, not organizer rules.

## At a glance

More Than a Hack is a football analytics hackathon organized by Mobile World Capital Barcelona, Barça Innovation Hub (BIHUB), and Metrica Sports. Teams use real FC Barcelona match data to develop sports analytics ideas.

The event has two online challenges followed by an on-site final pitch in Barcelona. The initial challenge asks teams to estimate team possession for every frame of a match using supplied automatic tracking data and homography. The innovation challenge is an open-ended analytics or tactical project for the top 20 teams.

### Important dates

| Phase | Published dates | Format / eligibility |
|---|---|---|
| Registration | October 1–November 10, 2026 | Online |
| Initial Challenge | October 20–November 17, 2026 | Online; all registered teams |
| Innovation Challenge | November 26, 2026–January 14, 2027 | Online; top 20 teams from the Initial Challenge |
| Final Pitch | March 2, 2027 | On-site at Talent Arena; top 5 teams from the Innovation Challenge, subject to jury selection |

**Deadline note:** The published registration close date is November 10, 2026. As of November 10, registration is listed as closing today. Check the official registration page promptly for the exact cutoff time and whether registration is still open.

## Who the event is for

The organizers invite football analysts, data analysts and scientists, software developers, and technology enthusiasts interested in using data and technology to help teams and athletes perform better.

## Challenge structure

### 1. Initial Challenge

- Open to all registered teams.
- Estimate **team possession for every frame** of a football match.
- Use provided automatic tracking data (ATD) and homography.
- The challenge is scoped so participants do not need to build detection, player identification, appearance, or calibration systems from scratch.
- Submissions are automatically evaluated and ranked.
- The top 20 teams advance to the Innovation Challenge.

The event page does **not** publish the submission interface, exact file format, possession label definition, evaluation metric, baseline, or detailed rules. Obtain these from the official challenge materials when available; do not assume a particular interpretation of possession.

### 2. Innovation Challenge

- Open to the top 20 teams from the Initial Challenge.
- Teams choose analytical or tactical questions and explore analysis, modeling, interpretation, or insight generation.
- The organizers say the selected teams will work with a dataset of **10 professional matches** under realistic analytical conditions, with specialized technical mentoring.
- The top 5 teams advance to the final pitch.

### 3. Final Pitch

- Scheduled for **March 2, 2027**, on-site at Talent Arena in Barcelona.
- The top 5 teams from the Innovation Challenge are invited to pitch; final selection is made by a professional jury.
- The page says travel and accommodation expenses to Barcelona for the final pitch are covered. Confirm the exact coverage, conditions, and booking process with the organizers before making plans.

## Team and registration requirements

- Each team must have **3 to 5 members**.
- Teams with fewer than 3 members are **not eligible** for the Innovation Challenge or Final Pitch.
- Each participant registers with personal details. The published form requests:
  - First and last name
  - ID/passport number
  - Country and city of residence
  - Email and phone number
  - LinkedIn profile (required)
  - GitHub or equivalent profile (optional on the displayed form)
  - Professional profile (select a category)
  - Team name
  - Acceptance of Mobile World Capital Barcelona's Privacy Policy and More Than a Hack Terms & Conditions
- The form states participant information will be used by the hackathon team to contact participants about the activity.
- Registration page: use the **Register now** link on the [official event page](https://landing.mobileworldcapital.com/more-than-a-hack-2027).

Agree the team name and membership before registering, and make sure every member completes the required registration steps. Review the Privacy Policy and Terms & Conditions directly before accepting them.

## Data and tools described by the organizers

The event page lists access to real datasets provided by FC Barcelona and Metrica Sports:

| Data / resource | Provider stated on the event page | Published description |
|---|---|---|
| Video recordings of official matches | Barça Innovation Hub | Match video |
| Automatic tracking data (ATD) | Metrica Sports | Tracking data; used with homography in the Initial Challenge |
| Automatic event data (Smart Tagging) | Metrica Sports | Automatically generated event data |
| Innovation dataset | Organizers / challenge program | 10 professional matches for top 20 teams, with technical mentoring |
| Analytics platform | Metrica Sports | Access highlighted by the event page; details and access terms to be confirmed |

The event page does not specify data schemas, coordinate systems, frame rates, synchronization procedures, permitted external data, licensing, or retention rules. Follow the challenge documentation and data-use terms provided to registered participants.

## Awards and participant benefits

The event page lists:

- Travel and accommodation expenses to Barcelona for the final pitch covered
- Official FC Barcelona 2026/27 shirts
- Official BIHUB merchandise
- An exclusive FC Barcelona experience

It does not describe prize quantities, eligibility details, travel limits, or whether every finalist receives every item. Confirm the terms with the organizers.

## Getting ready to participate

### Immediate checklist

- [ ] Check whether registration is still available on the official page; the listed close date is November 10, 2026.
- [ ] Confirm a team of 3–5 members and settle on one team name.
- [ ] Ensure every member completes the individual registration and required profile fields.
- [ ] Save the official terms, privacy policy, challenge rules, and any organizer announcements in the team’s shared reference folder.
- [ ] Confirm the challenge platform, submission deadline/time zone, file format, evaluation metric, and allowed tools as soon as challenge materials are released.
- [ ] Create a shared communication channel and agree on meeting times, task ownership, and decision-making.

### Initial Challenge preparation

Before data access, prepare to inspect the supplied files and documentation rather than assuming a schema. When materials arrive:

1. Read the task definition and determine exactly what “team possession” means for a frame, including how ambiguous, contested, dead-ball, or out-of-play frames are labeled.
2. Inspect tracking and homography fields, coordinate conventions, missing values, sampling rate, and any match/video alignment information.
3. Establish a simple baseline and a local validation approach that respects the official split and rules.
4. Track data quality and error cases; compare predictions against labels only where the challenge materials authorize this.
5. Package and validate the submission early, leaving time for the official submission system and final checks.

These are workflow suggestions, not a claim about the organizer’s hidden evaluation rules. Avoid using data or methods that the challenge rules prohibit.

## Repository setup recommendation

The organizer’s page does not mandate a programming language or repository layout. The following is a lightweight, adaptable starting point for a team repository:

```text
.
├── README.md                 # Project purpose, setup, and team workflow
├── MORE_THAN_A_HACK_2027.md  # Event facts and participation guide
├── .gitignore
├── docs/                     # Official rules, schema notes, decisions (no restricted data)
├── notebooks/                # Exploratory work; keep outputs manageable
├── src/                      # Reusable code
├── tests/                    # Small checks for parsing and core logic
├── configs/                  # Non-secret, non-sensitive configuration
├── submissions/              # Submission-generation code or permitted artifacts
└── data/                     # Local only; do not commit organizer-provided data
```

### Suggested setup steps

1. Create a private repository for the team unless the competition rules require another visibility setting.
2. Add all teammates and agree on a default branch and pull-request workflow.
3. Add a `README.md` with the project scope, environment setup, how to run the pipeline, and who owns each part.
4. Add a `.gitignore` for local environments, caches, notebook checkpoints, generated outputs, credentials, and `data/`.
5. Keep raw match videos, tracking data, and other restricted assets out of Git. Store them only in locations and formats allowed by the organizer’s terms.
6. Record package versions in the language’s standard dependency lock/manifest file once the team chooses its stack.
7. Keep analysis reproducible: document input assumptions, random seeds where relevant, and the command that creates a submission.
8. Never commit passwords, access tokens, private participant details, or challenge data.

### Team workflow suggestion

- Assign a challenge lead, data/analytics lead, and engineering/reproducibility lead; one person may cover multiple roles.
- Use small issues or tasks with an owner and a clear outcome.
- Keep experiments and decisions in concise notes so teammates can reproduce and review them.
- Share code through version control; share restricted data only through approved storage.
- Maintain a known-good baseline and a working submission path throughout the challenge.

## Questions and official links

- **Hackathon contact:** [hackathon.talentarena@mobileworldcapital.com](mailto:hackathon.talentarena@mobileworldcapital.com)
- **More Than a Hack 2027 page:** [landing.mobileworldcapital.com/more-than-a-hack-2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
- **Talent Arena site:** [talentarena.tech](https://talentarena.tech)

For anything not specified on the event page—including rules, challenge mechanics, exact deadlines, data access, mentoring, travel coverage, and prizes—ask the organizers or check the official participant materials.

## Source and scope

Event facts in this guide were checked against the [official More Than a Hack 2027 page](https://landing.mobileworldcapital.com/more-than-a-hack-2027) on November 10, 2026. This guide summarizes the public page and adds clearly marked practical recommendations; it is not a substitute for the official rules, terms, or participant instructions.
