---
layout: page
title: "History"
permalink: /history/
---

<style>
.filter-btns { margin-bottom: 1em; }
.filter-btn {
  cursor: pointer;
  padding: 4px 10px;
  margin: 3px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: none;
  font-size: 0.85em;
}
.filter-btn.active {
  background: #555;
  color: white;
  border-color: #555;
}
.history-table { overflow-x: auto; }
.history-table table { white-space: nowrap; width: 100%; border-collapse: separate; border-spacing: 0; }
.history-table td, .history-table th { padding: 6px 20px 6px 4px; }
.history-table td:last-child { white-space: normal; min-width: 200px; }
</style>

<div class="filter-btns">
  <button class="filter-btn active" onclick="filterHistory('all')">All</button>
  <button class="filter-btn" onclick="filterHistory('💻')">💻 Work</button>
  <button class="filter-btn" onclick="filterHistory('🔬')">🔬 Research</button>
  <button class="filter-btn" onclick="filterHistory('🧠')">🧠 Hackathon</button>
  <button class="filter-btn" onclick="filterHistory('🎓')">🎓 Education</button>
  <button class="filter-btn" onclick="filterHistory('👩‍🏫')">👩‍🏫 Teaching</button>
  <button class="filter-btn" onclick="filterHistory('🌐')">🌐 Event</button>
</div>

<div class="history-table" markdown="1">

| | Date | Location | Organization | Event |
|-|------|----------|--------------|-------|
| 🌐 | Jun 2026 | New York, NY | InstaLILY AI | InstaLILY x Women in AI 2026 NYC Tech Week Invitee |
| 🧠 | Dec 2025 | New York, NY | ElevenLabs | Track Winner, ElevenLabs 2025 Worldwide Conversational AI Agents Hackathon |
| 💻 | Aug 2025 – Present | New York, NY | Amazon | Software Engineer, Live Events @ Amazon Ads |
| 🔬 | May 2025 | Rotterdam, Netherlands | Interspeech 2025 | Paper accepted, Interspeech 2025 |
| 🔬 | May 2025 | Vienna, Austria | ACL 2025 | Paper accepted, ACL 2025 Findings: *Statement-Tuning Enables Efficient Cross-lingual Generalization* |
| 🔬 | Sep 2024 | Kos, Greece | Interspeech 2024 | Scholarship & presentation, Interspeech 2024 Young Female Researchers in Speech Workshop |
| 🔬 | Aug 2024 | Bangkok, Thailand | ACL 2024 | Travel Grant, Spotlight Paper, Oral & Poster Presentation in Main Conference, ACL 2024 Student Research Workshop |
| 💻 | May 2024 - Jun 2024 | Abu Dhabi, UAE | MBZUAI | Research Intern, Best Team Award in Undergraduate Research Internship Program |
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
