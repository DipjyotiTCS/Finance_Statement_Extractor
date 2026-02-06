
function formatConfidence(conf) {
  if (conf === null || conf === undefined) return "";

  // Preserve already-formatted percentages
  if (typeof conf === "string") {
    const s = conf.trim();
    if (!s) return "";
    if (s.endsWith("%")) return s;
    const n = parseFloat(s.replace(",", "."));
    if (!Number.isFinite(n)) return s;
    conf = n;
  }

  if (typeof conf !== "number" || !Number.isFinite(conf)) return String(conf);

  // Most of the pipeline uses 0..1. Display as percentage.
  if (conf <= 1) return Math.round(conf * 100) + "%";

  // If something already produced 0..100, treat as percent for display.
  if (conf <= 100) return Math.round(conf) + "%";

  return String(conf);
}

function renderTable(extracted) {
  const rows = extracted.rows || {};
  const head = document.getElementById("json-head");
  const body = document.getElementById("json-body");

  head.innerHTML = "";
  body.innerHTML = "";

  // discover year columns dynamically
  const years = new Set();
  Object.values(rows).forEach(r => {
    Object.keys(r).forEach(k => {
      if (/^\d{4}$/.test(k)) years.add(k);
    });
  });

  const yearCols = Array.from(years).sort((a,b) => b-a);

  // build header
  let headerRow = "<tr><th>Item</th>";
  yearCols.forEach(y => headerRow += `<th>${y}</th>`);
  headerRow += "<th>Note</th><th>Confidence</th><th>Action</th></tr>";
  head.innerHTML = headerRow;

  // build rows
  Object.entries(rows).forEach(([item, data]) => {
    let tr = document.createElement("tr");
    tr.innerHTML = `<td>${item}</td>`;

    yearCols.forEach(y => {
      tr.innerHTML += `<td contenteditable="false">${data[y] ?? ""}</td>`;
    });

    tr.innerHTML += `<td contenteditable="false">${data.nota ?? ""}</td>`;
    tr.innerHTML += `<td contenteditable="false">${formatConfidence(data.confidence_score ?? 0)}</td>`;
    tr.innerHTML += `<td><button class="edit-btn">Edit</button></td>`;

    body.appendChild(tr);
  });
}
