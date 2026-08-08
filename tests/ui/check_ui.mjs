// Checks that src/paybands/api/static/index.html really builds itself from the
// API, rather than that it *looks like* it does when you read the source.
//
//   uv run uvicorn paybands.api.app:app --port 8137 &
//   npm --prefix tests/ui install && node tests/ui/check_ui.mjs
//
// pytest covers the API; nothing covered the page, and the page is where the
// honesty is supposed to become visible. A hardcoded dropdown that has drifted
// from the model looks completely fine in code review — the only way to catch
// it is to build the DOM and compare it against a live /schema.
//
// It talks to a REAL running server for the payloads, then replays them into
// jsdom. Fixtures would defeat the point: the bug this exists to catch is the
// page and the model disagreeing, and two hand-written fixtures always agree.
import { JSDOM } from "jsdom";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PAGE = path.join(HERE, "..", "..", "src", "paybands", "api", "static", "index.html");
const BASE = process.env.PAYBANDS_URL || "http://127.0.0.1:8137";

const html = fs.readFileSync(PAGE, "utf8");

let schema, predict;
try {
  schema = await (await fetch(`${BASE}/schema`)).json();
  predict = await (await fetch(`${BASE}/predict-band`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ candidate: { years_experience: 5, city: "Bangalore", level: "Senior" } }),
  })).json();
} catch (e) {
  console.error(`Could not reach ${BASE} — start the server first.\n  ${e.message}`);
  process.exit(2);
}

const dom = new JSDOM(html, {
  runScripts: "dangerously", url: "http://127.0.0.1:8137/",
  beforeParse(win) {            // must exist BEFORE the page script runs
    win.fetch = async (path) => ({
      ok: true, status: 200,
      json: async () => (path === "/schema" ? schema : predict),
    });
    win.addEventListener("error", (e) => console.log("PAGE ERROR:", e.message));
  },
});
const w = dom.window;

await new Promise(r => w.addEventListener("load", r));
await new Promise(r => setTimeout(r, 400));

const $ = (id) => w.document.getElementById(id);
const fail = [];
const ok = (cond, msg) => cond ? console.log("  ✓ " + msg) : (fail.push(msg), console.log("  ✗ " + msg));

console.log("\nDropdowns built from /schema:");
for (const id of ["role","education","org_size","remote"]) {
  const got = [...$(id).options].slice(1).map(o => o.value);
  ok(JSON.stringify(got) === JSON.stringify(schema.categories[id]),
     `${id}: ${got.length} options, first = "${got[0]}"`);
}

console.log("\nInert fields marked:");
for (const id of schema.inert_fields) {
  const el = $(id); if (!el) { console.log(`  – ${id}: no form control (skipped)`); continue; }
  const label = el.closest("label");
  ok(label.classList.contains("inert") && label.querySelector(".badge-inert"), `${id} dimmed + badged`);
}
console.log("\nUsable fields NOT marked:");
for (const id of schema.usable_fields) {
  const el = $(id); if (!el) continue;
  ok(!el.closest("label").classList.contains("inert"), `${id} left normal`);
}

console.log("\nProvenance + notes:");
ok($("trained").textContent.includes(schema.trained_on), "banner names the training set");
ok($("role").value === "Developer, back-end" || schema.categories.role.includes($("role").value),
   `default role = "${$("role").value}" (exists in this bundle)`);
ok(!$("notescard").classList.contains("hidden"), "notes card visible when notes present");
const li = [...$("notes").querySelectorAll("li")].map(n => n.textContent);
ok(li.length === predict.notes.length + 1,
   `${li.length} notes: ${predict.notes.length} from the API + the UI's no-skills warning`);
ok(li[0].includes("cannot tell that apart from having none"),
   "no-skills warning leads, since it biases the number most");
ok(!$("out").classList.contains("hidden"), "results rendered on first load");
ok($("range").textContent.includes("₹"), `band shown: ${$("range").textContent}`);

console.log("\nSkills input:");
ok($("skills").querySelectorAll('input[type=checkbox]').length === schema.known_skills.length,
   `${schema.known_skills.length} skill checkboxes built from /schema`);
$("skills").querySelector('input[value="python"]').checked = true;
$("skills_other").value = "AWS, Docker";
w.$ = undefined;
const c = w.eval("candidate()");
ok(JSON.stringify(c.skills) === JSON.stringify(["python","AWS","Docker"]),
   `candidate().skills = ${JSON.stringify(c.skills)}`);
const empty = (() => { $("skills").querySelector('input[value="python"]').checked = false;
                       $("skills_other").value = ""; return w.eval("candidate()"); })();
ok(!("skills" in empty), "untouched form omits skills rather than sending []");

console.log(fail.length ? `\nFAILED: ${fail.length}` : "\nALL CHECKS PASSED");
process.exit(fail.length ? 1 : 0);
