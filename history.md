---
layout: page
title: "History"
permalink: /history/
---

<style>
.filter-btns { margin-bottom: 1.2em; display: flex; flex-wrap: wrap; gap: 6px; }
.filter-btn {
  cursor: pointer;
  padding: 5px 12px;
  border: 1px solid #dde3ea;
  border-radius: 999px;
  background: #fff;
  font-size: 0.82em;
  color: #444;
  transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
.filter-btn:hover { border-color: #555; color: #555; }
.filter-btn.active {
  background: #555;
  color: #fff;
  border-color: #555;
}

.history-table { overflow-x: auto; }
.history-table table {
  white-space: nowrap;
  width: 100%;
  border-collapse: separate;
  border-spacing: 0 10px;
}
.history-table thead th {
  padding: 4px 20px 8px 4px;
  font-size: 0.75em;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #999;
  border: none;
}
.history-table tbody tr:hover td { background: #f2f2f2; }
.history-table td {
  padding: 10px 20px 10px 4px;
  background: #fcfcfc;
  border-top: 1px solid #ececec;
  border-bottom: 1px solid #ececec;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  transition: background 0.15s ease;
}
.history-table td:last-child {
  white-space: normal;
  min-width: 200px;
  border-right: 1px solid #ececec;
  border-radius: 0 8px 8px 0;
}
.history-table td:nth-child(2) { font-weight: 600; color: #333; }
.history-table .duration { margin-top: 2px; font-weight: 400; color: #aaa; font-size: 0.78em; }
.history-table td:nth-child(3) { color: #888; font-size: 0.92em; }
.history-table td:nth-child(4) { color: #111; font-weight: 500; }
.org-continued { color: #aaa; font-weight: 400; font-size: 0.88em; }

/* timeline spine running through the icon column */
.history-table td:first-child {
  position: relative;
  width: 36px;
  text-align: center;
  border-left: 1px solid #ececec;
  border-radius: 8px 0 0 8px;
}
.history-table td:first-child::before {
  content: "";
  position: absolute;
  top: -10px;
  height: calc(50% + 10px);
  left: 50%;
  width: 2px;
  background: #dde3ea;
  transform: translateX(-50%);
  z-index: 0;
}
.history-table td:first-child::after {
  content: "";
  position: absolute;
  top: 50%;
  bottom: -10px;
  left: 50%;
  width: 2px;
  background: #dde3ea;
  transform: translateX(-50%);
  z-index: 0;
}
.history-table tbody tr:first-child td:first-child::before { content: none; }
.history-table tbody tr:last-child td:first-child::after { content: none; }

/* darken the connector specifically between roles at the same org */
.history-table tr.org-group:not(.org-group-first) td:first-child::before {
  background: #555;
  width: 3px;
}
.history-table tr.org-group:not(.org-group-last) td:first-child::after {
  background: #555;
  width: 3px;
}
.history-table .dot {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: #fff;
  border: 2px solid #555;
  font-size: 12px;
  line-height: 1;
}

/* stacked card layout on narrow viewports such as mobile devices */
@media screen and (max-width: 600px) {
  .history-table table,
  .history-table thead,
  .history-table tbody,
  .history-table tr,
  .history-table td {
    display: block;
    width: 100%;
  }
  .history-table thead { display: none; }
  .history-table tbody tr {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    column-gap: 10px;
    margin-bottom: 12px;
    padding: 12px 14px;
    background: #fcfcfc;
    border: 1px solid #ececec;
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  }
  .history-table tbody tr:hover { background: #f2f2f2; }
  .history-table td {
    padding: 0;
    border: none;
    box-shadow: none;
    background: transparent;
    white-space: normal;
    min-width: 0;
  }
  .history-table td:first-child {
    order: 1;
    width: auto;
    border-left: none;
    border-radius: 0;
  }
  .history-table td:first-child::before,
  .history-table td:first-child::after { content: none; }
  .history-table td:nth-child(2) {
    order: 2;
    flex: 1 1 auto;
  }
  .history-table td:nth-child(3) {
    order: 4;
    flex: 0 0 100%;
    margin-top: 4px;
  }
  .history-table td:nth-child(3)::before { content: "📍 "; }
  .history-table td:nth-child(4) {
    order: 3;
    flex: 0 0 100%;
    margin-top: 8px;
  }
  .history-table td:last-child {
    order: 5;
    flex: 0 0 100%;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #ececec;
    border-right: none;
    border-radius: 0;
  }
}
</style>

<div class="filter-btns">
  <button class="filter-btn active" onclick="filterHistory('all')">All</button>
  <button class="filter-btn" onclick="filterHistory('💻')">💻 Work</button>
  <button class="filter-btn" onclick="filterHistory('🔬')">🔬 Research</button>
  <button class="filter-btn" onclick="filterHistory('🧠')">🧠 Hackathon</button>
  <button class="filter-btn" onclick="filterHistory('🎓')">🎓 Education</button>
  <button class="filter-btn" onclick="filterHistory('👩‍🏫')">👩‍🏫 Teaching</button>
  <button class="filter-btn" onclick="filterHistory('🌐')">🌐 Event</button>
  <button class="filter-btn" onclick="filterHistory('🗣️')">🗣️ Language</button>
</div>

<div class="history-table" markdown="1">

| | Date | Location | Organization | Event |
|-|------|----------|--------------|-------|
| 🌐 | Jun 2026 | New York, NY | InstaLILY AI | NYC Tech Week 2026 Women in AI Invitee |
| 🧠 | Dec 2025 | New York, NY | ElevenLabs | Track Winner, ElevenLabs 2025 Worldwide Conversational AI Agents Hackathon |
| 💻 | Aug 2025 – Oct 2026 | New York, NY | Amazon | Software Engineer, Live Events @ Amazon Ads |
| 🔬 | May 2025 | Rotterdam, Netherlands | Interspeech 2025 | Paper accepted, Interspeech 2025 |
| 🔬 | May 2025 | Vienna, Austria | ACL 2025 | Paper accepted, ACL 2025 Findings: *Statement-Tuning Enables Efficient Cross-lingual Generalization* |
| 🔬 | Sep 2024 | Kos, Greece | Interspeech 2024 | Scholarship & presentation, Interspeech 2024 Young Female Researchers in Speech Workshop: *Generating Code-Switching Speech Based on Intonation Units for Non-English Language Pairs* |
| 🔬 | Aug 2024 | Bangkok, Thailand | ACL 2024 | Travel Grant, Spotlight Paper, Oral & Poster Presentation in Main Conference, ACL 2024 Student Research Workshop: *CoVoSwitch: Machine Translation of Synthetic Code-Switched Text Based on Intonation Units* |
| 💻 | May 2024 - Jun 2024 | Abu Dhabi, UAE | MBZUAI | Research Intern @ Natural Language Processing Department, Best Team Award in Undergraduate Research Internship Program |
| 💻 | Jan 2024 – Apr 2024 | South Korea | NAVER | Machine Learning Engineer Intern, Text-to-Speech @ Multimodal AI |
| 👩‍🏫 | Jan 2023 – Dec 2023 | New Haven, CT | Yale University | Teaching Assistant, CPSC 223 Data Structures and Programming Techniques |
| 💻 | Jun 2023 – Aug 2023 | South Korea | Samsung Electronics | Software Engineer, ML Intern, Text-to-Speech @ AI R&D |
| 👩‍🏫 | Aug 2022 – Dec 2022 | New Haven, CT | Yale University | Teaching Assistant, CS50 |
| 🧠 | Oct 2022 | Cambridge, MA | MIT | HackMIT 2022 Finalist |
| 👩‍🏫 | Sep 2021 – May 2022 | New Haven, CT | Code Haven at Yale | Mentor, Scratch programming |
| 💻 | Sep 2021 - Jan 2022 | New Haven, CT | Yale University | Tobin Undergraduate Research Assistant, Yale Department of Economics & School of Management |
| 🎓 | Aug 2020 – May 2025 | New Haven, CT | Yale University | Bachelor of Science, Computer Science and Economics |

</div>

<script>
document.querySelectorAll('.history-table tbody tr').forEach(row => {
  const cell = row.querySelector('td');
  if (cell) cell.innerHTML = '<span class="dot">' + cell.textContent.trim() + '</span>';
});

// group consecutive rows that share the same Organization and category into one visual tenure
const historyRows = Array.from(document.querySelectorAll('.history-table tbody tr'));
historyRows.forEach(row => {
  row.dataset.org = row.querySelectorAll('td')[3]?.textContent.trim();
  row.dataset.icon = row.querySelector('.dot')?.textContent.trim();
});
let gi = 0;
while (gi < historyRows.length) {
  const org = historyRows[gi].dataset.org;
  const icon = historyRows[gi].dataset.icon;
  let gEnd = gi + 1;
  while (gEnd < historyRows.length && historyRows[gEnd].dataset.org === org && historyRows[gEnd].dataset.icon === icon) gEnd++;
  if (gEnd - gi > 1) {
    for (let k = gi; k < gEnd; k++) {
      historyRows[k].classList.add('org-group');
      if (k === gi) historyRows[k].classList.add('org-group-first');
      if (k === gEnd - 1) historyRows[k].classList.add('org-group-last');
      if (k > gi) {
        const orgCell = historyRows[k].querySelectorAll('td')[3];
        if (orgCell) orgCell.innerHTML = '<span class="org-continued">↳ ' + org + '</span>';
      }
    }
  }
  gi = gEnd;
}

const MONTHS = {Jan:1,Feb:2,Mar:3,Apr:4,May:5,Jun:6,Jul:7,Aug:8,Sep:9,Oct:10,Nov:11,Dec:12};
document.querySelectorAll('.history-table tbody tr').forEach(row => {
  const icon = row.querySelector('.dot')?.textContent.trim();
  if (icon !== '💻') return;
  const dateCell = row.querySelectorAll('td')[1];
  if (!dateCell) return;
  const match = dateCell.textContent.trim().match(/^([A-Za-z]+)\s+(\d{4})\s*[-–—]\s*([A-Za-z]+)\s+(\d{4})$/);
  if (!match) return;
  const [, startMon, startYear, endMon, endYear] = match;
  if (!(startMon in MONTHS) || !(endMon in MONTHS)) return;
  const total = (parseInt(endYear, 10) * 12 + MONTHS[endMon]) - (parseInt(startYear, 10) * 12 + MONTHS[startMon]) + 1;
  if (total <= 0) return;
  const years = Math.floor(total / 12), months = total % 12;
  const parts = [];
  if (years) parts.push(years + (years > 1 ? ' yrs' : ' yr'));
  if (months) parts.push(months + ' mo');
  const durationEl = document.createElement('div');
  durationEl.className = 'duration';
  durationEl.textContent = parts.join(' ');
  dateCell.appendChild(durationEl);
});

function filterHistory(category) {
  const rows = document.querySelectorAll('.history-table table tbody tr');
  rows.forEach(row => {
    const icon = row.querySelector('td')?.textContent.trim();
    row.style.display = (category === 'all' || icon === category) ? '' : 'none';
  });
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.classList.toggle('active', category === 'all' ? btn.textContent === 'All' : btn.textContent.startsWith(category));
  });
}
</script>
