/*
 * charts.js
 * WHAT: Creates the Chart.js charts on the dashboard.
 * WHY:  Charts make year-to-year change easy to see.
 * HOW:  Both charts start EMPTY (with a "No analysis yet" overlay).
 *       Later stages call EarthCharts.updateTrend(...) and
 *       EarthCharts.updateLandUse(...) with real results.
 */
(function () {
    const COLORS = {
        vegetation: "#2e9e5b",
        water: "#1e6091",
        urban: "#e08a1e",
        other: "#9aa5a0",
    };

    let trendChart = null;
    let landChart = null;

    function showOverlay(id, show, message) {
        const el = document.getElementById(id);
        if (!el) return;
        if (message) el.textContent = message;
        el.style.display = show ? "flex" : "none";
    }

    function init() {
        const msg = "Chart library not found. Check that static/vendor/chart.umd.js exists.";
        if (typeof Chart === "undefined") {
            showOverlay("trend-empty", true, msg);
            showOverlay("landuse-empty", true, msg);
            return;
        }

        // Line chart: area of each land type for each year
        trendChart = new Chart(document.getElementById("trend-chart"), {
            type: "line",
            data: {
                labels: [],
                datasets: [
                    { label: "Vegetation (km²)", data: [], borderColor: COLORS.vegetation, backgroundColor: COLORS.vegetation, tension: 0.25 },
                    { label: "Water (km²)", data: [], borderColor: COLORS.water, backgroundColor: COLORS.water, tension: 0.25 },
                    { label: "Urban (km²)", data: [], borderColor: COLORS.urban, backgroundColor: COLORS.urban, tension: 0.25 },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, title: { display: true, text: "Area (km²)" } } },
            },
        });

        // Doughnut chart: share of each land type in one year
        landChart = new Chart(document.getElementById("landuse-chart"), {
            type: "doughnut",
            data: {
                labels: [],
                datasets: [{ data: [], backgroundColor: [COLORS.vegetation, COLORS.water, COLORS.urban, COLORS.other] }],
            },
            options: { responsive: true, maintainAspectRatio: false },
        });
    }

    /* data = { years: [2018,2022,2026], vegetation: [..], water: [..], urban: [..] } */
    function updateTrend(data) {
        if (!trendChart || !data || !Array.isArray(data.years) || data.years.length === 0) return;
        trendChart.data.labels = data.years;
        trendChart.data.datasets[0].data = data.vegetation || [];
        trendChart.data.datasets[1].data = data.water || [];
        trendChart.data.datasets[2].data = data.urban || [];
        trendChart.update();
        showOverlay("trend-empty", false);
    }

    /* data = { labels: ["Vegetation","Water","Urban","Other"], values: [..] } */
    function updateLandUse(data) {
        if (!landChart || !data || !Array.isArray(data.labels) || data.labels.length === 0) return;
        landChart.data.labels = data.labels;
        landChart.data.datasets[0].data = data.values || [];
        landChart.update();
        showOverlay("landuse-empty", false);
    }

    document.addEventListener("DOMContentLoaded", init);
    window.EarthCharts = { updateTrend: updateTrend, updateLandUse: updateLandUse };
})();
