import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [csvPath, metadataPath, outputPath, previewDataPath, previewMetaPath, inspectPath] = process.argv.slice(2);
if (!inspectPath) throw new Error("Expected six arguments.");

const csvText = await fs.readFile(csvPath, "utf8");
const metadata = JSON.parse(await fs.readFile(metadataPath, "utf8"));
const rows = metadata.qa.actuals.od_rows + 1;
const imported = await Workbook.fromCSV(csvText, { sheetName: "Imported" });
const importedSheet = imported.worksheets.getItem("Imported");
const workbook = Workbook.create();
const dataSheet = workbook.worksheets.add("LIGHT_DIRTY_OD_v01");
const used = dataSheet.getRange(`A1:H${rows}`);
dataSheet.getRange("A1:H1").values = [importedSheet.getRange("A1:H1").values[0]];
for (let start = 2; start <= rows; start += 2500) {
  const end = Math.min(rows, start + 2499);
  const count = end - start + 1;
  const source = importedSheet.getRangeByIndexes(start - 1, 0, count, 8).values;
  const converted = source.map((row) => [
    Number(row[0]), row[1], Number(row[2]), row[3],
    Number(row[4]), Number(row[5]), Number(row[6]), Number(row[7]),
  ]);
  dataSheet.getRangeByIndexes(start - 1, 0, count, 8).values = converted;
}

dataSheet.showGridLines = false;
dataSheet.freezePanes.freezeRows(1);
used.format.font = { name: "Arial", size: 10, color: "#1F1F1F" };
dataSheet.getRange("A1:H1").format = {
  fill: "#1F4E78",
  font: { name: "Arial", bold: true, color: "#FFFFFF", size: 10 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
dataSheet.getRange(`A2:A${rows}`).format.numberFormat = "0";
dataSheet.getRange(`C2:C${rows}`).format.numberFormat = "0";
dataSheet.getRange(`E2:H${rows}`).format.numberFormat = "0.000000000000";
dataSheet.getRange(`A1:A${rows}`).format.columnWidth = 16;
dataSheet.getRange(`B1:B${rows}`).format.columnWidth = 28;
dataSheet.getRange(`C1:C${rows}`).format.columnWidth = 20;
dataSheet.getRange(`D1:D${rows}`).format.columnWidth = 28;
dataSheet.getRange(`E1:H${rows}`).format.columnWidth = 18;
dataSheet.getRange("A1:H1").format.rowHeight = 24;

const meta = workbook.worksheets.add("Metadata_QA");
meta.showGridLines = false;
meta.getRange("A2:D2").values = [["LIGHT_DIRTY_OD_v01", null, null, null]];
meta.getRange("A3:D3").format.borders = { preset: "doubleBottom", style: "thin", color: "#1F4E78" };
meta.getRange("A5:B18").values = [
  ["Field", "Value"],
  ["Dataset label", metadata.dataset_label],
  ["Scale label", metadata.scale_label],
  ["Status", metadata.status],
  ["beta_dirty_demonstrator", metadata.beta_dirty_demonstrator],
  ["k_dirty_demonstrator", metadata.k_dirty_demonstrator],
  ["M1 k* anchor", metadata.m1_k_anchor],
  ["Q_v0 (veh/day)", metadata.q_v0_veh_day],
  ["Q_dirty noncommuting (veh/day)", metadata.q_dirty_noncommuting_veh_day],
  ["Method N", "N_dirty_ij = 0.15 * N_v0_ij"],
  ["Method T", "T_dirty_ij = C_ISTAT_ij + N_dirty_ij"],
  ["Interpretation", metadata.assumption],
  ["D1 original", metadata.historical_status.D1_original],
  ["D1-R1", metadata.historical_status.D1_R1],
];
const checkRows = Object.entries(metadata.qa.checks).map(([name, value]) => [name, value ? "PASS" : "FAIL"]);
meta.getRange("A21:B21").values = [["Mandatory QA", "Result"]];
meta.getRangeByIndexes(21, 0, checkRows.length, 2).values = checkRows;
meta.getRange("A2:D40").format.font = { name: "Arial", size: 10, color: "#1F1F1F" };
meta.getRange("A5:B5").format = {
  fill: "#1F4E78",
  font: { name: "Arial", bold: true, color: "#FFFFFF", size: 10 },
};
meta.getRange("A21:B21").format = {
  fill: "#1F4E78",
  font: { name: "Arial", bold: true, color: "#FFFFFF", size: 10 },
};
meta.getRange("A2").format.font = { name: "Arial", size: 14, bold: true, color: "#1F1F1F" };
meta.getRange("A:A").format.columnWidth = 34;
meta.getRange("B:B").format.columnWidth = 96;
meta.getRange("C:D").format.columnWidth = 3;
meta.getRange("B6:B18").format.wrapText = true;
meta.getRange("B9:B13").format.numberFormat = "0.000000000000";
meta.getRange("A2:B40").format.verticalAlignment = "center";
meta.getRange("A6:A40").format.font = { name: "Arial", size: 10, bold: true, color: "#1F1F1F" };
meta.getRange("A2:B40").format.autofitRows();
meta.getRange("A2:B40").format.rowHeight = 20;
meta.getRange("B16:B18").format.rowHeight = 36;

workbook.recalculate();
const dataInspect = await workbook.inspect({
  kind: "table",
  range: "LIGHT_DIRTY_OD_v01!A1:H6",
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 8,
  maxChars: 5000,
});
const metaInspect = await workbook.inspect({
  kind: "table",
  range: "Metadata_QA!A2:B35",
  include: "values,formulas",
  tableMaxRows: 35,
  tableMaxCols: 2,
  maxChars: 8000,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  inspectPath,
  JSON.stringify({ data: dataInspect.ndjson, metadata: metaInspect.ndjson, errors: errors.ndjson }, null, 2) + "\n",
  "utf8",
);
const previewData = await workbook.render({ sheetName: "LIGHT_DIRTY_OD_v01", autoCrop: "all", scale: 1.4, format: "png" });
const previewMeta = await workbook.render({ sheetName: "Metadata_QA", autoCrop: "all", scale: 1.4, format: "png" });
await fs.writeFile(previewDataPath, new Uint8Array(await previewData.arrayBuffer()));
await fs.writeFile(previewMetaPath, new Uint8Array(await previewMeta.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
