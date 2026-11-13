/*
 * change.js
 * WHAT: The Change Detection page: before/after slider, change map, tables and bars.
 * WHY:  Lets the user SEE what changed between two years and read the numbers.
 * HOW:  Loads the saved report from GET /api/change/<analysis id> (made by the dashboard's
 *       Analyze Region), then redraws everything when the user picks another pair of years.
 */
(function () {
    const root = document.getElementById("cd-root");
    if (!root) return;

    const VIEW_HINTS = {
        rgb: "True colour picture. Drag the slider to compare the two years.",
        ndvi: "NDVI: dark green = dense vegetation, brown/tan = little or none.",
        ndwi: "NDWI: blue = open water, brown = dry land.",
        ndbi: "NDBI: orange/red = built-up or bare surfaces, green = vegetation.",
        landcover: "Land cover: green = vegetation, orange = built-up, blue = water, grey = bare / other.",
    };

    let report = null;
    let pairIndex = 0;
    let view = "rgb";
    let mapKind = "change";

    function el(tag, text, className) {
        const node = document.createElement(tag);
        if (text !== undefined && text !== null) node.textContent = text;
        if (className) node.className = className;
        return node;
    }

    function addRow(table, cells, isHeader, numericFrom) {
        const row = el("tr");
        cells.forEach(function (c, i) {
            const cell = el(isHeader ? "th" : "td", c);
            if (numericFrom !== undefined && i >= numericFrom) cell.className = "num";
            row.appendChild(cell);
        });
        table.appendChild(row);
    }

    function pct(v) { return typeof v === "number" ? (v > 0 ? "+" : "") + v.toFixed(1) + "%" : "n/a"; }
    function signed(v, digits) { return (v > 0 ? "+" : "") + v.toFixed(digits); }

    // ---------- before / after slider ----------
    const compare = document.getElementById("cmp");
    const range = document.getElementById("cmp-range");

    function setSlider(percent) {
        percent = Math.max(0, Math.min(100, percent));
        range.value = percent;
        document.getElementById("cmp-before-wrap").style.clipPath = "inset(0 " + (100 - percent) + "% 0 0)";
        document.getElementById("cmp-handle").style.left = percent + "%";
    }
    range.addEventListener("input", function () { setSlider(parseFloat(range.value)); });

    let dragging = false;
    function dragTo(event) {
        const box = compare.getBoundingClientRect();
        setSlider((event.clientX - box.left) / box.width * 100);
    }
    compare.addEventListener("pointerdown", function (e) { dragging = true; dragTo(e); });
    window.addEventListener("pointermove", function (e) { if (dragging) dragTo(e); });
    window.addEventListener("pointerup", function () { dragging = false; });

    function showCompare() {
        const pair = report.pairs[pairIndex];
        const before = report.year_pictures[String(pair.from_year)][view];
        const after = report.year_pictures[String(pair.to_year)][view];
        document.getElementById("cmp-before").src = before;
        document.getElementById("cmp-after").src = after;
        document.getElementById("cmp-tag-before").textContent = pair.from_year + " (before)";
        document.getElementById("cmp-tag-after").textContent = pair.to_year + " (after)";
        document.getElementById("cd-compare-title").textContent = pair.label;
        document.getElementById("cd-view-hint").textContent = VIEW_HINTS[view];
    }

    // ---------- change map ----------
    function showMap() {
        const pair = report.pairs[pairIndex];
        document.getElementById("cd-map-img").src = pair.pictures[mapKind];
        const legend = document.getElementById("cd-map-legend");
        legend.replaceChildren();
        if (mapKind === "change") {
            legend.appendChild(el("div", "Change map: " + pair.label, "legend-title"));
            const chips = el("div", null, "legend-chips");
            report.legends.change.forEach(function (c) {
                const chip = el("span", null, "chip");
                const dot = document.createElement("i");
                dot.style.background = c.color;
                chip.appendChild(dot);
                chip.appendChild(document.createTextNode(c.name));
                chips.appendChild(chip);
            });
            legend.appendChild(chips);
        } else {
            const d = report.legends.difference;
            legend.appendChild(el("div", d.title, "legend-title"));
            const bar = el("div", null, "legend-bar");
            bar.style.background = "linear-gradient(to right, " + d.stops.map(function (s) {
                return s.color + " " + s.percent + "%";
            }).join(", ") + ")";
            legend.appendChild(bar);
            const ticks = el("div", null, "legend-ticks");
            ticks.appendChild(el("span", String(d.min) + " (less green)"));
            ticks.appendChild(el("span", "+" + String(d.max) + " (more green)"));
            legend.appendChild(ticks);
        }
    }

    // ---------- numbers ----------
    function showNumbers() {
        const pair = report.pairs[pairIndex];
        document.getElementById("cd-pair-label").textContent = pair.label;
        document.getElementById("cd-summary").textContent =
            "Compared over " + pair.analysed_km2.toFixed(1) + " km² (ground usable in every year). " +
            pair.changed_km2.toFixed(2) + " km² (" + pair.changed_percent.toFixed(1) +
            " %) changed class. Mean NDVI changed by " + signed(pair.mean_ndvi_change, 3) + ".";

        const table = document.getElementById("cd-table");
        table.replaceChildren();
        addRow(table, ["Class", pair.from_year + " (km²)", pair.to_year + " (km²)", "Net change (km²)",
                       "Net change (%)", "Gained (km²)", "Lost (km²)"], true, 1);
        pair.categories.forEach(function (c) {
            addRow(table, [c.name, c.area_from_km2.toFixed(3), c.area_to_km2.toFixed(3), signed(c.net_km2, 3),
                           pct(c.net_percent), c.gained_km2.toFixed(3), c.lost_km2.toFixed(3)], false, 1);
        });

        // bars
        const bars = document.getElementById("cd-bars");
        bars.replaceChildren();
        const maxAbs = Math.max.apply(null, pair.categories.map(function (c) { return Math.abs(c.net_km2); }).concat([0.001]));
        pair.categories.forEach(function (c) {
            const row = el("div", null, "bar-row");
            row.appendChild(el("span", c.name));
            const track = el("div", null, "bar-track");
            const fill = el("div", null, "bar-fill " + (c.net_km2 >= 0 ? "up" : "down"));
            fill.style.width = (Math.abs(c.net_km2) / maxAbs * 50) + "%";
            track.appendChild(fill);
            row.appendChild(track);
            row.appendChild(el("span", signed(c.net_km2, 3) + " km²", "bar-value"));
            bars.appendChild(row);
        });

        // transitions
        const trans = document.getElementById("cd-trans");
        trans.replaceChildren();
        addRow(trans, ["From", "To", "Area (km²)", "Share of analysed ground"], true, 2);
        if (pair.transitions.length === 0) {
            addRow(trans, ["No change patches above the noise level", "", "", ""], false);
        }
        pair.transitions.slice(0, 10).forEach(function (t) {
            addRow(trans, [t.from, t.to, t.km2.toFixed(3), t.percent_of_area.toFixed(2) + " %"], false, 2);
        });

        // reliability
        const warnBox = document.getElementById("cd-warnings");
        warnBox.replaceChildren();
        if (pair.warnings.length) {
            const list = el("ul", null, "warning-list");
            pair.warnings.forEach(function (w) { list.appendChild(el("li", w)); });
            warnBox.appendChild(list);
        }
        const m = pair.noise_margins;
        document.getElementById("cd-margins").textContent =
            "Normal date-to-date differences measured on unchanged ground: NDVI ±" + m.NDVI.toFixed(2) +
            ", NDWI ±" + m.NDWI.toFixed(2) + ", NDBI ±" + m.NDBI.toFixed(2) +
            ". Class changes smaller than this are ignored as noise.";
        const notes = document.getElementById("cd-notes");
        notes.replaceChildren();
        report.notes.forEach(function (n) { notes.appendChild(el("li", n, "step-why")); });
    }

    function selectPair(index) {
        pairIndex = index;
        document.querySelectorAll("#cd-pairs .seg-btn").forEach(function (b, i) {
            b.classList.toggle("active", i === index);
        });
        showCompare();
        showMap();
        showNumbers();
    }

    // ---------- controls ----------
    document.querySelectorAll("#cd-views .seg-btn").forEach(function (button) {
        button.addEventListener("click", function () {
            view = button.dataset.view;
            document.querySelectorAll("#cd-views .seg-btn").forEach(function (b) { b.classList.toggle("active", b === button); });
            if (report) showCompare();
        });
    });
    document.querySelectorAll("#cd-maps .seg-btn").forEach(function (button) {
        button.addEventListener("click", function () {
            mapKind = button.dataset.map;
            document.querySelectorAll("#cd-maps .seg-btn").forEach(function (b) { b.classList.toggle("active", b === button); });
            if (report) showMap();
        });
    });

    function showEmpty(message) {
        document.getElementById("cd-loading").hidden = true;
        document.getElementById("cd-main").hidden = true;
        if (message) document.getElementById("cd-empty-text").textContent = message;
        document.getElementById("cd-empty").hidden = false;
    }

    async function load() {
        const id = root.dataset.analysis;
        if (!id) { showEmpty(); return; }
        try {
            const response = await fetch("/api/change/" + id);
            const payload = await response.json();
            if (!payload.ok) { showEmpty(payload.error); return; }
            report = payload;

            const pairs = document.getElementById("cd-pairs");
            pairs.replaceChildren();
            report.pairs.forEach(function (pair, i) {
                const button = el("button", pair.label, "seg-btn");
                button.type = "button";
                button.addEventListener("click", function () { selectPair(i); });
                pairs.appendChild(button);
            });
            document.getElementById("cd-region-hint").textContent =
                "Analysis #" + report.analysis_id + ". Years: " + report.years.join(", ") +
                ". Land cover from: " + (report.method_label || "Rule-based (index thresholds)") +
                ". The last button compares the first and last year (the overall change).";

            document.getElementById("cd-loading").hidden = true;
            document.getElementById("cd-main").hidden = false;
            setSlider(50);
            selectPair(report.pairs.length - 1);      // start with the overall change
        } catch (error) {
            showEmpty("Could not load the change results. Make sure python app.py is running, then try again.");
        }
    }
    load();
})();
