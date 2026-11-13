/*
 * dashboard.js
 * WHAT: Behaviour of the dashboard page.
 * WHY:  Check inputs early, talk to the Flask API, show the dataset summary.
 * HOW:  - "Sample region": POST /api/prepare/sample
 *       - "My own region": check the 3 files here, then POST them to /api/prepare/upload
 *       - Result: a dataset table + the real area drawn on the map.
 *       STAGE 4: after the dataset is ready, the page calls /api/preprocess/<id>
 *       and shows the preprocessing report (cleaned pictures, cloud mask, steps).
 *       STAGE 5: then it calls /api/indices/<id> and shows NDVI / NDWI / NDBI pictures,
 *       the rule-based land cover, a table of areas (km2) and fills the two charts.
 *       STAGE 7: if "Machine learning" is chosen it calls /api/ml/<id> (trained classifier,
 *       compared with the rules) and then measures change on the ML land cover.
 *       STAGE 6: finally it calls /api/change/<id>, fills the four Earth Health cards
 *       (change between the first and last year) and a "Change over time" table.
 *       The Change Detection page (static/js/change.js) shows the before/after maps.
 *       EarthUI.setHealthCards / setDiagnosis are used by later stages.
 */
(function () {
    const ALLOWED_EXT = [".tif", ".tiff"];
    const MAX_MB = 80;   // keep the same as MAX_FILE_MB in config.py

    const sourceSelect = document.getElementById("data-source");
    const nameRow = document.getElementById("custom-name-row");
    const nameInput = document.getElementById("region-name");
    const slots = document.querySelectorAll(".upload-slot");
    const hint = document.getElementById("source-hint");
    const form = document.getElementById("analyze-form");
    const statusBox = document.getElementById("status-box");
    const analyzeBtn = document.getElementById("analyze-btn");

    // ---------- small helpers ----------
    /* Create an element. We use textContent so user text can never run as HTML. */
    function el(tag, text, className) {
        const node = document.createElement(tag);
        if (text !== undefined && text !== null) node.textContent = text;
        if (className) node.className = className;
        return node;
    }

    function showStatus(type, text) {
        statusBox.className = "status-box " + type;
        statusBox.textContent = text;
        statusBox.hidden = false;
    }

    function setBusy(busy, text) {
        analyzeBtn.disabled = busy;
        analyzeBtn.textContent = busy ? (text || "⏳ Working...") : "🔍 Analyze Region";
    }

    /* Call an API and return its JSON. Returns null if the reply was not JSON. */
    async function callApi(url, options) {
        const response = await fetch(url, options);
        try { return { payload: await response.json(), status: response.status }; }
        catch (e) { return { payload: null, status: response.status }; }
    }

    function getExt(filename) {
        const dot = filename.lastIndexOf(".");
        return dot === -1 ? "" : filename.slice(dot).toLowerCase();
    }

    /* Returns an error message, or "" if the file is fine. */
    function checkFile(file) {
        if (!ALLOWED_EXT.includes(getExt(file.name))) {
            return "Only GeoTIFF files (.tif / .tiff) are accepted.";
        }
        if (file.size === 0) return "This file is empty.";
        if (file.size > MAX_MB * 1024 * 1024) {
            return "File is larger than " + MAX_MB + " MB. Crop a smaller area.";
        }
        return "";
    }

    function formatProblems(payload) {
        let text = payload.error || "Something went wrong.";
        if (payload.problems && payload.problems.length) {
            text += "\n• " + payload.problems.join("\n• ");
        }
        return text;
    }

    // ---------- switch between sample / custom ----------
    function updateSourceMode() {
        const isSample = sourceSelect.value === "sample";
        nameRow.hidden = isSample;
        slots.forEach(function (s) { s.classList.toggle("disabled", isSample); });
        hint.textContent = isSample
            ? "Sample region reads the real Sentinel-2 band files in data/sample/demo_region/ (one folder per year: 2018, 2019, 2026). Upload slots are off in this mode."
            : "Upload one 5-band Sentinel-2 GeoTIFF per year (bands B02, B03, B04, B08, B11). Use the same area for every year.";
    }
    sourceSelect.addEventListener("change", updateSourceMode);
    updateSourceMode();

    // ---------- show file name / errors in each slot ----------
    slots.forEach(function (slot) {
        const input = slot.querySelector(".file-input");
        const info = slot.querySelector(".slot-info");
        input.addEventListener("change", function () {
            slot.classList.remove("has-file", "has-error");
            info.classList.remove("error");
            if (input.files.length === 0) {
                info.textContent = "No file chosen";
                return;
            }
            const file = input.files[0];
            const problem = checkFile(file);
            if (problem) {
                slot.classList.add("has-error");
                info.classList.add("error");
                info.textContent = problem;
            } else {
                slot.classList.add("has-file");
                info.textContent = file.name + " (" + (file.size / 1048576).toFixed(1) + " MB)";
            }
        });
    });

    // ---------- build the upload form (returns FormData, or null after showing an error) ----------
    function buildUploadForm() {
        if (nameInput.value.trim() === "") {
            showStatus("error", "Please enter a region name.");
            return null;
        }
        const formData = new FormData();
        formData.append("region_name", nameInput.value.trim());

        const years = [];
        for (let i = 1; i <= 3; i++) {
            const yearValue = parseInt(document.getElementById("year" + i).value, 10);
            const fileInput = document.getElementById("file" + i);
            if (isNaN(yearValue) || yearValue < 2015 || yearValue > 2035) {
                showStatus("error", "Slot " + i + ": enter a valid year (2015 to 2035).");
                return null;
            }
            if (fileInput.files.length === 0) {
                showStatus("error", "Slot " + i + " (" + yearValue + "): please choose a GeoTIFF file.");
                return null;
            }
            const problem = checkFile(fileInput.files[0]);
            if (problem) {
                showStatus("error", "Slot " + i + " (" + yearValue + "): " + problem);
                return null;
            }
            years.push(yearValue);
            formData.append("year" + i, yearValue);
            formData.append("file" + i, fileInput.files[0]);
        }
        if (new Set(years).size !== years.length) {
            showStatus("error", "Each slot needs a different year.");
            return null;
        }
        return formData;
    }

    // ---------- show the dataset summary ----------
    function addRow(table, cells, isHeader) {
        const row = el("tr");
        cells.forEach(function (c) { row.appendChild(el(isHeader ? "th" : "td", c)); });
        table.appendChild(row);
    }

    function renderDataset(result) {
        const card = document.getElementById("dataset-card");
        const table = document.getElementById("dataset-table");
        table.replaceChildren();

        addRow(table, ["Year", "Source", "Size (pixels)", "Bands", "Coordinate system",
                       "Pixel size", "Area covered", "Valid pixels", "Value scale"], true);

        result.years.forEach(function (item) {
            const m = item.meta;
            let source = item.source_kind === "folder" ? "Band files merged: " : "GeoTIFF: ";
            source += item.original_filename;
            if (item.scene && item.scene.acquisition_date) {
                source += " (scene date " + item.scene.acquisition_date + ")";
            }
            addRow(table, [
                String(item.year),
                source,
                m.width + " × " + m.height,
                String(m.bands),
                m.crs || "none",
                m.pixel_width_m ? m.pixel_width_m.toFixed(1) + " m" : "n/a",
                m.area_km2 ? m.area_km2.toFixed(1) + " km²" : "n/a",
                m.valid_pixel_percent.toFixed(1) + " %",
                m.value_scale,
            ], false);
        });

        // Band explanation table
        const bandTable = document.getElementById("band-table");
        bandTable.replaceChildren();
        addRow(bandTable, ["Band", "Name", "Wavelength", "Native resolution", "Used for"], true);
        result.band_order.forEach(function (band) {
            const b = result.band_info[band];
            addRow(bandTable, [band, b.name, "~" + b.wavelength_nm + " nm", b.resolution_m + " m", b.used_for], false);
        });

        // Warnings
        const warnBox = document.getElementById("dataset-warnings");
        warnBox.replaceChildren();
        if (result.warnings && result.warnings.length) {
            const list = el("ul", null, "warning-list");
            result.warnings.forEach(function (w) { list.appendChild(el("li", w)); });
            warnBox.appendChild(list);
        }

        document.getElementById("dataset-badge").textContent =
            result.mode === "sample" ? "Sample region · real Sentinel-2 data" : "Your uploaded files";
        document.getElementById("dataset-summary").textContent =
            "Region: " + result.region_name + " · analysis record #" + result.analysis_id +
            " · " + result.years.length + " years prepared. Nothing has been analysed yet.";
        card.hidden = false;

        // Draw the real area of the first year on the map
        const bounds = result.years[0].meta.bounds_latlon;
        if (bounds && window.EarthMap) {
            window.EarthMap.showAOI(bounds, result.region_name);
        }
    }

    // ---------- show the preprocessing report ----------
    function renderPreprocessing(report) {
        const card = document.getElementById("prep-card");
        const g = report.grid;

        document.getElementById("prep-summary").textContent =
            "All years now share one grid of " + g.width + " × " + g.height + " pixels (" +
            g.area_km2.toFixed(1) + " km²). " + g.common_usable_percent.toFixed(1) +
            " % of it is usable in every year. Nothing has been classified yet.";

        // Pictures (switchable: true colour / mask)
        const gallery = document.getElementById("prep-gallery");
        gallery.replaceChildren();
        report.years.forEach(function (item) {
            const box = el("figure", null, "prep-item");
            const img = document.createElement("img");
            img.alt = "Satellite picture of " + item.year;
            img.dataset.rgb = item.rgb_url;
            img.dataset.mask = item.mask_url;
            img.src = item.rgb_url;
            box.appendChild(img);
            box.appendChild(el("figcaption", String(item.year)));
            gallery.appendChild(box);
        });
        document.querySelectorAll("#prep-toggle .seg-btn").forEach(function (b) {
            b.classList.toggle("active", b.dataset.view === "rgb");
        });

        // Table
        const table = document.getElementById("prep-table");
        table.replaceChildren();
        addRow(table, ["Year", "Original size", "Aligned", "Valid pixels", "Cloud-like", "Usable"], true);
        report.years.forEach(function (y) {
            addRow(table, [
                String(y.year), y.native_size + " px", y.resampled ? "resampled" : "already aligned",
                y.valid_percent.toFixed(1) + " %", y.cloud_percent.toFixed(2) + " %", y.usable_percent.toFixed(1) + " %",
            ], false);
        });

        // Warnings
        const warnBox = document.getElementById("prep-warnings");
        warnBox.replaceChildren();
        if (report.warnings && report.warnings.length) {
            const list = el("ul", null, "warning-list");
            report.warnings.forEach(function (w) { list.appendChild(el("li", w)); });
            warnBox.appendChild(list);
        }

        // Steps with the reason for each
        const steps = document.getElementById("prep-steps");
        steps.replaceChildren();
        report.steps.forEach(function (s) {
            const li = el("li", null, s.applied ? "" : "skipped");
            li.appendChild(el("strong", s.name.replace(/^\d+\.\s*/, "") + (s.applied ? "" : " (skipped)")));
            li.appendChild(el("div", "Why: " + s.why, "step-why"));
            li.appendChild(el("div", s.detail, "step-detail"));
            steps.appendChild(li);
        });

        card.hidden = false;
    }

    // switch between true colour and mask pictures
    document.querySelectorAll("#prep-toggle .seg-btn").forEach(function (button) {
        button.addEventListener("click", function () {
            document.querySelectorAll("#prep-toggle .seg-btn").forEach(function (b) { b.classList.remove("active"); });
            button.classList.add("active");
            document.querySelectorAll("#prep-gallery img").forEach(function (img) {
                img.src = button.dataset.view === "mask" ? img.dataset.mask : img.dataset.rgb;
            });
        });
    });

    // ---------- show the indices (Stage 5) ----------
    let indexReport = null;

    function renderLegend(view) {
        const box = document.getElementById("ind-legend");
        box.replaceChildren();
        if (!indexReport || view === "rgb") return;
        const legend = indexReport.legends.find(function (l) { return l.key === view; });
        if (!legend) return;
        box.appendChild(el("div", legend.title, "legend-title"));
        if (legend.classes) {
            const chips = el("div", null, "legend-chips");
            legend.classes.forEach(function (c) {
                const chip = el("span", null, "chip");
                const dot = document.createElement("i");
                dot.style.background = c.color;
                chip.appendChild(dot);
                chip.appendChild(document.createTextNode(c.name));
                chips.appendChild(chip);
            });
            box.appendChild(chips);
        } else {
            const bar = el("div", null, "legend-bar");
            bar.style.background = "linear-gradient(to right, " + legend.stops.map(function (s) {
                return s.color + " " + s.percent + "%";
            }).join(", ") + ")";
            box.appendChild(bar);
            const ticks = el("div", null, "legend-ticks");
            ticks.appendChild(el("span", String(legend.min) + " (low)"));
            ticks.appendChild(el("span", String(legend.max) + " (high)"));
            box.appendChild(ticks);
        }
    }

    function showView(view) {
        document.querySelectorAll("#ind-toggle .seg-btn").forEach(function (b) {
            b.classList.toggle("active", b.dataset.view === view);
        });
        document.querySelectorAll("#ind-gallery img").forEach(function (img) {
            img.src = img.dataset[view];
        });
        renderLegend(view);
    }

    document.querySelectorAll("#ind-toggle .seg-btn").forEach(function (button) {
        button.addEventListener("click", function () { showView(button.dataset.view); });
    });

    function renderIndices(report) {
        indexReport = report;
        const last = report.years[report.years.length - 1];

        document.getElementById("ind-summary").textContent =
            "Measured over " + last.analysed_km2.toFixed(1) + " km² (the ground that is usable in every year). " +
            "These are estimates from index thresholds, not ground truth.";

        // Formula cards
        const formulas = document.getElementById("ind-formulas");
        formulas.replaceChildren();
        report.formulas.forEach(function (f) {
            const card = el("div", null, "formula-card");
            card.appendChild(el("strong", f.name));
            card.appendChild(el("code", f.formula));
            card.appendChild(el("div", "Bands: " + f.bands, "bands"));
            card.appendChild(el("div", f.meaning));
            formulas.appendChild(card);
        });

        // Pictures, one per year
        const gallery = document.getElementById("ind-gallery");
        gallery.replaceChildren();
        report.years.forEach(function (item) {
            const box = el("figure", null, "prep-item");
            const img = document.createElement("img");
            img.alt = "Index picture of " + item.year;
            Object.keys(item.pictures).forEach(function (key) { img.dataset[key] = item.pictures[key]; });
            box.appendChild(img);
            box.appendChild(el("figcaption", String(item.year)));
            gallery.appendChild(box);
        });
        showView("ndvi");

        // Table
        const table = document.getElementById("ind-table");
        table.replaceChildren();
        addRow(table, ["Year", "Mean NDVI", "Mean NDWI", "Mean NDBI", "Vegetation km²", "Water km²", "Built-up km²", "Bare / other km²"], true);
        report.years.forEach(function (y) {
            addRow(table, [
                String(y.year), y.mean_ndvi.toFixed(3), y.mean_ndwi.toFixed(3), y.mean_ndbi.toFixed(3),
                y.class_km2["Vegetation"].toFixed(2), y.class_km2["Water"].toFixed(3),
                y.class_km2["Built-up"].toFixed(2), y.class_km2["Bare / other"].toFixed(2),
            ], false);
        });

        // Warnings, thresholds, notes
        const warnBox = document.getElementById("ind-warnings");
        warnBox.replaceChildren();
        if (report.warnings && report.warnings.length) {
            const list = el("ul", null, "warning-list");
            report.warnings.forEach(function (w) { list.appendChild(el("li", w)); });
            warnBox.appendChild(list);
        }
        const thresholds = document.getElementById("ind-thresholds");
        thresholds.replaceChildren();
        Object.keys(report.thresholds).forEach(function (name) {
            const li = el("li");
            li.appendChild(el("strong", name + ": "));
            li.appendChild(document.createTextNode(report.thresholds[name]));
            thresholds.appendChild(li);
        });
        const notes = document.getElementById("ind-notes");
        notes.replaceChildren();
        report.notes.forEach(function (n) { notes.appendChild(el("li", n, "step-why")); });

        document.getElementById("indices-card").hidden = false;

        // Fill the two charts with the real numbers
        if (window.EarthCharts) {
            const round = function (v) { return Math.round(v * 1000) / 1000; };
            window.EarthCharts.updateTrend({
                years: report.years.map(function (y) { return y.year; }),
                vegetation: report.years.map(function (y) { return round(y.class_km2["Vegetation"]); }),
                water: report.years.map(function (y) { return round(y.class_km2["Water"]); }),
                urban: report.years.map(function (y) { return round(y.class_km2["Built-up"]); }),
            });
            window.EarthCharts.updateLandUse({
                labels: ["Vegetation", "Water", "Built-up", "Bare / other"],
                values: [last.class_km2["Vegetation"], last.class_km2["Water"], last.class_km2["Built-up"], last.class_km2["Bare / other"]].map(round),
            });
            document.getElementById("landuse-year").textContent = "(" + last.year + ", km²)";
        }
    }

    // ---------- machine-learning land cover (Stage 7) ----------
    let mlReport = null;

    function showMlView(view) {
        document.querySelectorAll("#ml-toggle .seg-btn").forEach(function (b) {
            b.classList.toggle("active", b.dataset.view === view);
        });
        document.querySelectorAll("#ml-gallery img").forEach(function (img) { img.src = img.dataset[view]; });

        const box = document.getElementById("ml-legend");
        box.replaceChildren();
        if (!mlReport) return;
        if (view === "confidence") {
            const legend = mlReport.confidence_legend;
            box.appendChild(el("div", "Model confidence (probability of the chosen class)", "legend-title"));
            const bar = el("div", null, "legend-bar");
            bar.style.background = "linear-gradient(to right, " + legend.stops.map(function (s) {
                return s.color + " " + s.percent + "%";
            }).join(", ") + ")";
            box.appendChild(bar);
            const ticks = el("div", null, "legend-ticks");
            ticks.appendChild(el("span", "30% or less (unsure)"));
            ticks.appendChild(el("span", "100% (sure)"));
            box.appendChild(ticks);
        } else if (indexReport) {
            const legend = indexReport.legends.find(function (l) { return l.key === "landcover"; });
            if (legend) {
                box.appendChild(el("div", view === "ml" ? "Land cover (machine learning)" : "Land cover (rule-based)", "legend-title"));
                const chips = el("div", null, "legend-chips");
                legend.classes.forEach(function (c) {
                    const chip = el("span", null, "chip");
                    const dot = document.createElement("i");
                    dot.style.background = c.color;
                    chip.appendChild(dot);
                    chip.appendChild(document.createTextNode(c.name));
                    chips.appendChild(chip);
                });
                box.appendChild(chips);
            }
        }
    }

    document.querySelectorAll("#ml-toggle .seg-btn").forEach(function (button) {
        button.addEventListener("click", function () { showMlView(button.dataset.view); });
    });

    function renderML(report) {
        mlReport = report;
        const model = report.model;
        const last = report.years[report.years.length - 1];

        document.getElementById("ml-badge").textContent = model.kind_label;
        document.getElementById("ml-summary").textContent =
            "A classifier learned what each land class looks like in all five bands from CONFIDENT pixels, then classified every pixel. " +
            "It agrees with the rule-based map on " + last.agreement_percent.toFixed(1) + " % of the ground in " + last.year +
            ". Agreement is not accuracy: with no ground-truth map, neither method is known to be right.";

        const gallery = document.getElementById("ml-gallery");
        gallery.replaceChildren();
        report.years.forEach(function (item) {
            const box = el("figure", null, "prep-item");
            const img = document.createElement("img");
            img.alt = "Land cover of " + item.year;
            Object.keys(item.pictures).forEach(function (key) { img.dataset[key] = item.pictures[key]; });
            box.appendChild(img);
            box.appendChild(el("figcaption", String(item.year)));
            gallery.appendChild(box);
        });

        const classes = ["Vegetation", "Built-up", "Bare / other", "Water"];
        const table = document.getElementById("ml-table");
        table.replaceChildren();
        const header = ["Year", "Agrees with rules", "Mean confidence", "Uncertain pixels"];
        classes.forEach(function (c) { header.push(c + " km² (rules → ML)"); });
        addRow(table, header, true);
        report.years.forEach(function (y) {
            const cells = [String(y.year), y.agreement_percent.toFixed(1) + " %", y.mean_confidence.toFixed(2), y.low_confidence_percent.toFixed(1) + " %"];
            classes.forEach(function (c) { cells.push(y.rule_km2[c].toFixed(2) + " → " + y.ml_km2[c].toFixed(2)); });
            addRow(table, cells, false);
        });

        document.getElementById("ml-matrix-year").textContent = String(last.year);
        const matrix = document.getElementById("ml-matrix");
        matrix.replaceChildren();
        const labels = last.agreement_matrix.labels;
        addRow(matrix, ["Rules ↓   Machine learning →"].concat(labels), true);
        last.agreement_matrix.percent.forEach(function (row, i) {
            const tr = el("tr");
            tr.appendChild(el("td", labels[i], "label-cell"));
            row.forEach(function (v, j) { tr.appendChild(el("td", v.toFixed(2) + " %", i === j ? "diag" : "")); });
            matrix.appendChild(tr);
        });

        const about = document.getElementById("ml-about");
        about.replaceChildren();
        function item(title, text) {
            const li = el("li");
            li.appendChild(el("strong", title + ": "));
            li.appendChild(document.createTextNode(text));
            about.appendChild(li);
        }
        item("Model", model.kind_label + ", trained " + model.trained_at + " on " + model.trained_on.source + " (years " + model.trained_on.years.join(", ") + ")");
        item("Learned classes", Object.keys(model.classes).map(function (k) { return model.classes[k]; }).join(", "));
        Object.keys(model.excluded_classes).forEach(function (name) { item("Not learned: " + name, model.excluded_classes[name]); });
        item("Training samples", Object.keys(model.training_samples).map(function (k) { return k + " " + model.training_samples[k]; }).join(", "));
        item("How pixels were labelled (confident only)", Object.keys(model.label_rules).map(function (k) { return k + ": " + model.label_rules[k]; }).join("; "));
        item("Hold-out test", (model.holdout.accuracy * 100).toFixed(1) + " % agreement on " + model.holdout.n_test + " held-back confident pixels");
        if (model.leave_one_year_out) {
            item("Leave-one-year-out test", (model.leave_one_year_out.accuracy * 100).toFixed(1) + " % (trained on " + model.leave_one_year_out.train_years.join(", ") + ", tested on " + model.leave_one_year_out.test_year + ")");
        }
        item("K-Means cross-check (no labels)", model.kmeans_check.agreement_percent.toFixed(1) + " % of pixels fall in a cluster that matches the rule-based class");
        item("Important", model.caveat);
        report.notes.forEach(function (n) { about.appendChild(el("li", n, "step-why")); });

        document.getElementById("ml-card").hidden = false;
        showMlView("ml");
    }

    // Is a trained model available? Shown next to the classification choice.
    (async function checkModel() {
        const badge = document.getElementById("ml-model-badge");
        try {
            const reply = await callApi("/api/ml/model");
            if (reply.payload && reply.payload.trained) {
                badge.textContent = "Model ready: " + reply.payload.info.kind_label;
                badge.className = "badge badge-green";
                document.getElementById("opt-method-ml").checked = true;
            } else {
                badge.textContent = "Not trained yet: run python train_model.py";
                badge.className = "badge badge-amber";
            }
        } catch (e) {
            badge.textContent = "Model status unavailable";
        }
    })();

    // ---------- show the change summary (Stage 6) ----------
    function percentText(v) {
        return typeof v === "number" ? (v > 0 ? "+" : "") + v.toFixed(1) + "%" : "n/a";
    }

    function renderChange(report) {
        const methodText = report.method === "ml" ? "machine learning" : "rule-based";
        setHealthCards(report.health, report.overall_label + " · " + methodText);
        document.getElementById("change-badge").textContent = report.method === "ml" ? "Machine-learning estimate" : "Rule-based estimate";

        const overall = report.pairs[report.pairs.length - 1];
        document.getElementById("change-summary").textContent =
            "Net change in the area of each class between the years. Overall (" + report.overall_label + "): " +
            overall.changed_percent.toFixed(1) + " % of the ground changed class (" + overall.changed_km2.toFixed(2) +
            " km² of " + overall.analysed_km2.toFixed(1) + " km²). \"n/a\" means the class was too small at the start for a reliable percentage.";

        const table = document.getElementById("change-table");
        table.replaceChildren();
        addRow(table, ["Years", "Vegetation", "Water", "Built-up", "Bare / other", "Ground that changed"], true);
        report.pairs.forEach(function (pair) {
            const by = {};
            pair.categories.forEach(function (c) { by[c.name] = c; });
            addRow(table, [
                pair.label,
                percentText(by["Vegetation"].net_percent), percentText(by["Water"].net_percent),
                percentText(by["Built-up"].net_percent), percentText(by["Bare / other"].net_percent),
                pair.changed_percent.toFixed(1) + " %",
            ], false);
        });

        const warnBox = document.getElementById("change-warnings");
        warnBox.replaceChildren();
        const warnings = [];
        report.pairs.forEach(function (pair) {
            pair.warnings.forEach(function (w) { warnings.push(pair.label + ": " + w); });
        });
        if (warnings.length) {
            const list = el("ul", null, "warning-list");
            warnings.forEach(function (w) { list.appendChild(el("li", w)); });
            warnBox.appendChild(list);
        }
        document.getElementById("change-link").href = "/change-detection?analysis=" + report.analysis_id;
        document.getElementById("change-card").hidden = false;
    }

    // ---------- Analyze button ----------
    form.addEventListener("submit", async function (event) {
        event.preventDefault();   // stay on the page
        statusBox.hidden = true;
        document.getElementById("prep-card").hidden = true;
        document.getElementById("indices-card").hidden = true;
        document.getElementById("change-card").hidden = true;
        document.getElementById("ml-card").hidden = true;

        let url, options;
        if (sourceSelect.value === "sample") {
            url = "/api/prepare/sample";
            options = { method: "POST" };
        } else {
            const formData = buildUploadForm();
            if (!formData) return;
            url = "/api/prepare/upload";
            options = { method: "POST", body: formData };
        }

        setBusy(true, "⏳ Reading satellite files...");
        try {
            // Step A: find, check and save the dataset (Stage 3)
            const prepared = await callApi(url, options);
            if (!prepared.payload) {
                showStatus("error", "The server sent an unexpected reply (HTTP " + prepared.status +
                    "). Check the terminal running app.py.");
                return;
            }
            if (!prepared.payload.ok) {
                showStatus("error", formatProblems(prepared.payload));
                return;
            }
            renderDataset(prepared.payload);

            // Step B: clean the images (Stage 4)
            setBusy(true, "⏳ Cleaning images (clouds, alignment)...");
            const cleaned = await callApi("/api/preprocess/" + prepared.payload.analysis_id, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mask_clouds: document.getElementById("opt-clouds").checked,
                    denoise: document.getElementById("opt-denoise").checked,
                }),
            });
            if (!cleaned.payload) {
                showStatus("error", "Preprocessing failed: unexpected server reply (HTTP " + cleaned.status +
                    "). Check the terminal running app.py.");
                return;
            }
            if (!cleaned.payload.ok) {
                showStatus("error", "The dataset is ready, but " + formatProblems(cleaned.payload));
                return;
            }
            renderPreprocessing(cleaned.payload);

            // Step C: indices and rule-based land cover (Stage 5)
            setBusy(true, "⏳ Calculating NDVI, NDWI, NDBI...");
            const indexed = await callApi("/api/indices/" + prepared.payload.analysis_id, { method: "POST" });
            if (!indexed.payload) {
                showStatus("error", "Index calculation failed: unexpected server reply (HTTP " + indexed.status +
                    "). Check the terminal running app.py.");
                return;
            }
            if (!indexed.payload.ok) {
                showStatus("error", "The images were cleaned, but " + formatProblems(indexed.payload));
                return;
            }
            renderIndices(indexed.payload);

            // Step C2: machine-learning classification (Stage 7), only if chosen
            const analysisId = prepared.payload.analysis_id;
            let method = document.getElementById("opt-method-ml").checked ? "ml" : "rules";
            let methodNote = "";
            if (method === "ml") {
                setBusy(true, "⏳ Machine-learning classification...");
                const ml = await callApi("/api/ml/" + analysisId, { method: "POST" });
                if (ml.payload && ml.payload.ok) {
                    renderML(ml.payload);
                } else {
                    method = "rules";
                    methodNote = " Machine learning was skipped (" +
                        (ml.payload ? ml.payload.error : "unexpected server reply") + ") so rule-based results were used.";
                }
            }

            // Step D: change between the years (Stage 6)
            setBusy(true, "⏳ Measuring change between years...");
            const changed = await callApi("/api/change/" + analysisId, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ method: method }),
            });
            if (!changed.payload) {
                showStatus("error", "Change detection failed: unexpected server reply (HTTP " + changed.status +
                    "). Check the terminal running app.py.");
                return;
            }
            if (!changed.payload.ok) {
                showStatus("error", "The indices were calculated, but " + formatProblems(changed.payload));
                return;
            }
            renderChange(changed.payload);
            showStatus(methodNote ? "info" : "success",
                "Analysis finished (record #" + prepared.payload.analysis_id +
                "): data checked, images cleaned, indices calculated and change measured (" +
                (method === "ml" ? "machine-learning" : "rule-based") + " land cover). " +
                "Open the Change Detection page for before/after maps." + methodNote);
        } catch (error) {
            showStatus("error",
                "Could not reach the server. Make sure python app.py is still running. " +
                "(Uploads larger than " + MAX_MB + " MB per file are refused.)");
        } finally {
            setBusy(false);
        }
    });

    // ---------- functions used by later stages ----------
    /*
     * values = { vegetation: 14.3, water: null, urban: 7.6, cropstress: 1.7 }
     *   vegetation / water / urban = net area change in percent (null = too little to measure)
     *   cropstress = % of the starting vegetation that is weakened (card is called "Vegetation stress")
     * Colours: vegetation/water going down is bad, urban growth is a warning, stress is bad when high.
     */
    function setHealthCards(values, labelText) {
        const cards = {
            vegetation: { cls: function (v) { return v < 0 ? "bad" : "good"; }, sub: "Net area change" },
            water: { cls: function (v) { return v < 0 ? "bad" : "good"; }, sub: "Net area change" },
            urban: { cls: function (v) { return v > 0 ? "warn" : "good"; }, sub: "Net area change" },
            cropstress: { cls: function (v) { return v >= 15 ? "bad" : (v >= 5 ? "warn" : "good"); }, sub: "of vegetation weakened" },
        };
        Object.keys(cards).forEach(function (key) {
            const card = document.getElementById("card-" + key);
            const valueEl = document.getElementById("val-" + key);
            if (!card || !valueEl || !(key in values)) return;
            const v = values[key];
            card.classList.remove("good", "bad", "warn");
            if (typeof v !== "number") {
                valueEl.textContent = "n/a";
                card.querySelector(".stat-sub").textContent = "Too little to measure";
                return;
            }
            card.classList.add(cards[key].cls(v));
            valueEl.textContent = key === "cropstress"
                ? v.toFixed(1) + "%"
                : (v < 0 ? "↓ " : "↑ ") + Math.abs(v).toFixed(1) + "%";
            card.querySelector(".stat-sub").textContent = cards[key].sub;
        });
        const label = document.getElementById("data-label");
        if (label && labelText) {
            label.textContent = labelText;
            label.className = "badge badge-blue";
        }
    }

    /* risk = "low" | "moderate" | "high" */
    function setDiagnosis(text, risk) {
        const badge = document.getElementById("risk-badge");
        const cls = { low: "badge-green", moderate: "badge-amber", high: "badge-red" };
        document.getElementById("diagnosis-text").textContent = text;
        badge.textContent = risk.charAt(0).toUpperCase() + risk.slice(1);
        badge.className = "badge " + (cls[risk] || "badge-grey");
    }

    window.EarthUI = { setHealthCards: setHealthCards, setDiagnosis: setDiagnosis };
})();
