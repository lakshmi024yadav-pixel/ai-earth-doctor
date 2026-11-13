/*
 * map.js
 * WHAT: Shows the Leaflet map on the dashboard.
 * WHY:  The study area and later the change/risk/suitability overlays
 *       are easiest to understand on a map.
 * HOW:  Base map = OpenStreetMap tiles (needs internet). The map is
 *       centred on Gurugram as a starting view only. Later stages call
 *       EarthMap.showAOI(...) with the real bounds read from the
 *       Sentinel-2 files.
 */
(function () {
    const DEFAULT_CENTER = [28.45, 77.03];   // Gurugram (start view only)
    const DEFAULT_ZOOM = 10;

    let map = null;
    let aoiLayer = null;

    function init() {
        const el = document.getElementById("map");
        if (!el) return;

        if (typeof L === "undefined") {
            el.innerHTML = '<div class="map-msg">Map library not found. Check that static/vendor/leaflet.js exists.</div>';
            return;
        }

        map = L.map("map").setView(DEFAULT_CENTER, DEFAULT_ZOOM);

        const tiles = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
            maxZoom: 19,
            attribution: "&copy; OpenStreetMap contributors",
        }).addTo(map);

        // If there is no internet, tell the user once, don't crash.
        let warned = false;
        tiles.on("tileerror", function () {
            if (warned) return;
            warned = true;
            const note = document.getElementById("map-note");
            if (note) {
                note.textContent = "Base map tiles could not load (no internet?). The app still works; only the street background is missing.";
            }
        });
    }

    /* bounds = [[south, west], [north, east]] */
    function showAOI(bounds, label) {
        if (!map || !bounds) return;
        if (aoiLayer) map.removeLayer(aoiLayer);
        aoiLayer = L.rectangle(bounds, { color: "#1f7a4d", weight: 2, fillOpacity: 0.08 }).addTo(map);
        if (label) aoiLayer.bindTooltip(label);
        map.fitBounds(bounds);
    }

    document.addEventListener("DOMContentLoaded", init);
    window.EarthMap = { showAOI: showAOI };
})();
