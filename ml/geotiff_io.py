"""
geotiff_io.py
--------------
WHAT:   Reads and writes GeoTIFF files (satellite images that know where on Earth they are).
WHY:    The project must run on every laptop. Rasterio, the usual library for this,
        contains compiled files that Windows "Smart App Control" can block.
        This module uses `tifffile`, which is pure Python, so nothing can be blocked.
HOW:    A GeoTIFF is a normal TIFF image plus a few extra tags:
          ModelPixelScale + ModelTiepoint   -> where pixel (0, 0) is and how big a pixel is
          GeoKeyDirectory                   -> the coordinate system (an EPSG number)
          GDAL_NODATA                       -> the value that means "no data"
        We read those tags, and write them again when we save a new file.
SUPPORTS: north-up images in WGS84 lat/lon (EPSG:4326), UTM on WGS84 (EPSG:326xx / 327xx) and
        Web Mercator (EPSG:3857). Sentinel-2 downloads from Copernicus Browser use one of these.
LIMITS: rotated images and other coordinate systems give a clear error message.
        Some compressions (LZW, JPEG) need the optional `imagecodecs` package; the Deflate
        compression used by Copernicus Browser works without it.
"""

import math

import cv2
import numpy as np
import tifffile

# TIFF / GeoTIFF tag numbers
TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_MODEL_TRANSFORM = 34264
TAG_GEO_KEYS = 34735
TAG_NODATA = 42113

# WGS84 ellipsoid
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
UTM_K0 = 0.9996


# ---------------------------------------------------------------------
# Small helper objects
# ---------------------------------------------------------------------
class Transform:
    """
    Pixel position -> map coordinates for a north-up image:
        x = c + column * a        y = f + row * e        (e is negative: rows go south)
    b and d (rotation) are always 0 here.
    """

    def __init__(self, a, c, e, f):
        self.a, self.b, self.c, self.d, self.e, self.f = float(a), 0.0, float(c), 0.0, float(e), float(f)

    def __eq__(self, other):
        if not isinstance(other, Transform):
            return False
        tol = abs(self.a) * 1e-3          # a thousandth of a pixel
        return (abs(self.a - other.a) <= tol and abs(self.e - other.e) <= tol
                and abs(self.c - other.c) <= tol and abs(self.f - other.f) <= tol)

    def __repr__(self):
        return "Transform(a=%r, c=%r, e=%r, f=%r)" % (self.a, self.c, self.e, self.f)


class Crs:
    """A coordinate system known by its EPSG number."""

    def __init__(self, epsg):
        self.epsg = int(epsg)

    @property
    def kind(self):
        if self.epsg in (4326, 4269, 4258):
            return "geographic"
        if 32601 <= self.epsg <= 32660 or 32701 <= self.epsg <= 32760:
            return "utm"
        if self.epsg in (3857, 900913):
            return "webmercator"
        return None                       # not supported

    @property
    def is_geographic(self):
        return self.kind == "geographic"

    @property
    def is_projected(self):
        return self.kind in ("utm", "webmercator")

    @property
    def linear_units(self):
        return "metre" if self.is_projected else "degree"

    def to_string(self):
        return "EPSG:%d" % self.epsg

    def __eq__(self, other):
        return isinstance(other, Crs) and self.epsg == other.epsg

    def __repr__(self):
        return "Crs(%s)" % self.to_string()


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------
def _epsg_from_geokeys(keys):
    """Read the EPSG number (and raster type) from the GeoKeyDirectory tag."""
    if not keys or len(keys) < 4:
        return None, 1
    entries = {}
    for i in range(int(keys[3])):
        key, location, _count, value = keys[4 + 4 * i: 8 + 4 * i]
        if location == 0:                  # the value is stored right here
            entries[key] = value
    raster_type = entries.get(1025, 1)     # 1 = pixel is an area (normal), 2 = pixel is a point
    for key in (3072, 2048):               # projected CS, then geographic CS
        if entries.get(key) not in (None, 0, 32767):
            return int(entries[key]), raster_type
    return None, raster_type


def _tag_value(page, code):
    tag = page.tags.get(code)
    return None if tag is None else tag.value


def read_geotiff(path):
    """
    Read a GeoTIFF. Returns a dict:
        data       numpy array, shape (bands, height, width), original number type
        transform  Transform, or None if the file has no georeferencing
        crs        Crs, or None if the file has no (known) coordinate system
        nodata     the "no data" value, or None
    Raises ValueError with a readable message if the file cannot be used.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if len(tif.pages) > 1 and page.samplesperpixel == 1:
                data = np.stack([p.asarray() for p in tif.pages])          # one band per page
            else:
                data = page.asarray()
                if data.ndim == 2:
                    data = data[np.newaxis]
                elif int(page.planarconfig) == 1:                          # (height, width, bands)
                    data = np.moveaxis(data, -1, 0)
            scale = _tag_value(page, TAG_PIXEL_SCALE)
            tiepoint = _tag_value(page, TAG_TIEPOINT)
            matrix = _tag_value(page, TAG_MODEL_TRANSFORM)
            geo_keys = _tag_value(page, TAG_GEO_KEYS)
            nodata_text = _tag_value(page, TAG_NODATA)
    except Exception as error:
        if "imagecodecs" in str(error):
            raise ValueError(
                "this TIFF uses a compression that needs the optional 'imagecodecs' package. "
                "Export it again without compression (or with Deflate)")
        raise ValueError("not a readable TIFF image (%s)" % str(error)[:120])

    transform = None
    if scale and tiepoint:
        sx, sy = float(scale[0]), float(scale[1])
        i, j, _, x, y, _ = [float(v) for v in tiepoint[:6]]
        transform = Transform(sx, x - i * sx, -sy, y + j * sy)
    elif matrix and len(matrix) >= 8:
        if abs(matrix[1]) > 1e-12 or abs(matrix[4]) > 1e-12:
            raise ValueError("the image is rotated, which is not supported. Export a north-up GeoTIFF")
        transform = Transform(matrix[0], matrix[3], matrix[5], matrix[7])

    epsg, raster_type = _epsg_from_geokeys(geo_keys)
    crs = Crs(epsg) if (epsg and transform) else None
    if transform and raster_type == 2:     # PixelIsPoint: the tiepoint is a pixel CENTRE, move to the corner
        transform = Transform(transform.a, transform.c - transform.a / 2, transform.e, transform.f - transform.e / 2)

    nodata = None
    if nodata_text:
        try:
            nodata = float(str(nodata_text).strip("\x00 \n"))
        except ValueError:
            nodata = None
    return {"data": data, "transform": transform, "crs": crs, "nodata": nodata}


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------
def write_geotiff(path, data, transform, crs, nodata=None):
    """Save a (bands, height, width) array as a Deflate-compressed GeoTIFF with its georeferencing."""
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError("data must have the shape (bands, height, width)")
    if crs is None or transform is None:
        raise ValueError("a GeoTIFF needs a transform and a coordinate system")

    model_type = 2 if crs.is_geographic else 1                    # 2 = geographic, 1 = projected
    cs_key = 2048 if crs.is_geographic else 3072
    keys = (1, 1, 0, 3, 1024, 0, 1, model_type, 1025, 0, 1, 1, cs_key, 0, 1, crs.epsg)
    extratags = [
        (TAG_PIXEL_SCALE, "d", 3, (abs(transform.a), abs(transform.e), 0.0), True),
        (TAG_TIEPOINT, "d", 6, (0.0, 0.0, 0.0, transform.c, transform.f, 0.0), True),
        (TAG_GEO_KEYS, "H", len(keys), keys, True),
    ]
    if nodata is not None:
        nodata = float(nodata)
        if math.isnan(nodata):
            text = "nan"
        elif nodata.is_integer():
            text = str(int(nodata))                  # "0", not "0.0" (integer images need this form)
        else:
            text = repr(nodata)
        extratags.append((TAG_NODATA, "s", 0, text, True))

    if data.shape[0] == 1:                           # a single band is a plain 2-D image
        tifffile.imwrite(path, data[0], photometric="minisblack", compression="zlib",
                         extratags=extratags, metadata=None)
    else:                                            # several bands: one plane per band
        tifffile.imwrite(path, data, photometric="minisblack", planarconfig="separate",
                         compression="zlib", extratags=extratags, metadata=None)


# ---------------------------------------------------------------------
# Where is it on Earth? (map coordinates -> latitude / longitude)
# ---------------------------------------------------------------------
def _utm_to_latlon(easting, northing, zone, south):
    """UTM (WGS84) -> (latitude, longitude) in degrees. Krueger series, accurate to well under 1 mm."""
    n = WGS84_F / (2 - WGS84_F)
    big_a = WGS84_A / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    beta = (n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96, n ** 2 / 48 + n ** 3 / 15, 17 * n ** 3 / 480)
    delta = (2 * n - 2 * n ** 2 / 3 - 2 * n ** 3, 7 * n ** 2 / 3 - 8 * n ** 3 / 5, 56 * n ** 3 / 15)

    xi = (northing - (10000000.0 if south else 0.0)) / (UTM_K0 * big_a)
    eta = (easting - 500000.0) / (UTM_K0 * big_a)
    xi_p = xi - sum(beta[j - 1] * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j in (1, 2, 3))
    eta_p = eta - sum(beta[j - 1] * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j in (1, 2, 3))
    chi = math.asin(math.sin(xi_p) / math.cosh(eta_p))
    lat = chi + sum(delta[j - 1] * math.sin(2 * j * chi) for j in (1, 2, 3))
    lon = math.radians(zone * 6 - 183) + math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return math.degrees(lat), math.degrees(lon)


def to_latlon(crs, x, y):
    """Map coordinates (x, y) -> (latitude, longitude). Raises ValueError for unsupported systems."""
    if crs.kind == "geographic":
        return y, x
    if crs.kind == "webmercator":
        lon = math.degrees(x / WGS84_A)
        lat = math.degrees(2 * math.atan(math.exp(y / WGS84_A)) - math.pi / 2)
        return lat, lon
    if crs.kind == "utm":
        south = crs.epsg >= 32701
        zone = crs.epsg - (32700 if south else 32600)
        return _utm_to_latlon(x, y, zone, south)
    raise ValueError("the coordinate system %s is not supported. Export with WGS84 (EPSG:4326) or UTM" % crs.to_string())


def bounds_latlon(transform, crs, width, height):
    """Bounding box of the image as (south, west, north, east) in degrees."""
    left, right = transform.c, transform.c + transform.a * width
    top, bottom = transform.f, transform.f + transform.e * height
    lats, lons = [], []
    for x in (left, (left + right) / 2, right):           # corners and edge middles
        for y in (top, (top + bottom) / 2, bottom):
            lat, lon = to_latlon(crs, x, y)
            lats.append(lat)
            lons.append(lon)
    return min(lats), min(lons), max(lats), max(lons)


def pixel_size_m(crs, transform, lat_mid):
    """Size of one pixel in metres (width, height), or (None, None) if the system is not supported."""
    if crs.kind == "geographic":
        return (abs(transform.a) * 111320.0 * math.cos(math.radians(lat_mid)), abs(transform.e) * 110570.0)
    if crs.kind == "utm":
        return abs(transform.a), abs(transform.e)
    if crs.kind == "webmercator":                          # stretched away from the equator
        scale = math.cos(math.radians(lat_mid))
        return abs(transform.a) * scale, abs(transform.e) * scale
    return None, None


def shrink(data, max_size=512):
    """Smaller copy of a (bands, height, width) array (area averaging), for quick statistics."""
    bands, height, width = data.shape
    if max(height, width) <= max_size:
        return data
    ratio = max_size / float(max(height, width))
    size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    return np.stack([cv2.resize(b.astype("float32"), size, interpolation=cv2.INTER_AREA) for b in data])
