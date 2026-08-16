"""Build the MRE demo scene — runs INSIDE Blender's embedded Python.

    blender.exe --background --python blender/build_scene.py -- <scene.json> <out.blend>

Reads the compact scene description written by ``scripts/demo.py`` (plain
JSON: buildings, roads, hospitals, population, intervention targets — no
geopandas, no GeoPackage, nothing Blender cannot read on its own) and builds
three side-by-side city blocks:

    BEFORE (normal city)  ->  AFTER (earthquake scenario)  ->  AFTER INTERVENTION

Buildings are coloured/shortened by damage state, roads coloured by link
state, hospitals and population units marked distinctly, and the intervention
block additionally highlights the entities the best-ranked portfolio targets.
A legend and title text are added, then a camera and a light are placed to
frame the whole layout before saving (and, if the render engine cooperates,
rendering a PNG preview).

Never imported by ``mre`` — ``bpy`` exists only in Blender's own interpreter.
This script performs no science: every colour and height multiplier it uses
comes from ``scene.json["visual"]``, produced by ``scripts/demo.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy  # noqa: F401 -- provided by Blender, not by the project venv
from mathutils import Vector

BLOCK_TITLES = ("BEFORE — NORMAL CITY", "AFTER — EARTHQUAKE SCENARIO", "AFTER INTERVENTION")


def parse_args() -> tuple[Path, Path]:
    if "--" not in sys.argv:
        raise SystemExit("usage: blender --background --python build_scene.py -- <scene.json> <out.blend>")
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 2:
        raise SystemExit("expected: <scene.json> <out.blend>")
    return Path(args[0]).resolve(), Path(args[1]).resolve()


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #

_material_cache: dict[tuple, "bpy.types.Material"] = {}


def get_material(name: str, color: tuple[float, float, float], emission: float = 0.0):
    key = (name, tuple(color), emission)
    if key in _material_cache:
        return _material_cache[key]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    rgba = (*color, 1.0)
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = rgba
            bsdf.inputs["Emission Strength"].default_value = emission
    mat.diffuse_color = rgba
    _material_cache[key] = mat
    return mat


# --------------------------------------------------------------------------- #
# geometry builders — all raw vertex/face data, the extruded-footprint pattern
# --------------------------------------------------------------------------- #


def add_box(name: str, collection, cx: float, cy: float, cz: float, sx: float, sy: float, sz: float, material):
    """Axis-aligned box centred at (cx, cy, base cz), extending up by sz."""
    hx, hy = sx / 2.0, sy / 2.0
    verts = [
        (cx - hx, cy - hy, cz), (cx + hx, cy - hy, cz), (cx + hx, cy + hy, cz), (cx - hx, cy + hy, cz),
        (cx - hx, cy - hy, cz + sz), (cx + hx, cy - hy, cz + sz), (cx + hx, cy + hy, cz + sz), (cx - hx, cy + hy, cz + sz),
    ]
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def add_segment_box(name: str, collection, x1, y1, x2, y2, width, height, material):
    """A thin box running from (x1,y1) to (x2,y2) — used for roads."""
    direction = Vector((x2 - x1, y2 - y1, 0.0))
    length = direction.length
    if length <= 1e-9:
        return None
    direction.normalize()
    perp = Vector((-direction.y, direction.x, 0.0)) * (width / 2.0)
    p1, p2 = Vector((x1, y1, 0.0)), Vector((x2, y2, 0.0))
    base = [p1 - perp, p2 - perp, p2 + perp, p1 + perp]
    top = [v + Vector((0, 0, height)) for v in base]
    verts = [tuple(v) for v in base + top]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def add_sphere(name: str, collection, x, y, z, radius, material):
    """A low-poly icosphere via bmesh — a lightweight population marker."""
    import bmesh

    mesh_obj_data = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=radius)
    bm.to_mesh(mesh_obj_data)
    bm.free()
    obj = bpy.data.objects.new(name, mesh_obj_data)
    obj.location = (x, y, z)
    collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def add_hospital_marker(name: str, collection, x, y, z, size, material):
    """A pole with a medical cross on top — reads as "hospital" at a glance."""
    pole = add_box(f"{name}_pole", collection, x, y, z, size * 0.25, size * 0.25, size * 1.6, material)
    top_z = z + size * 1.6
    add_box(f"{name}_cross_h", collection, x, y, top_z, size * 1.0, size * 0.28, size * 0.28, material)
    add_box(f"{name}_cross_v", collection, x, y, top_z, size * 0.28, size * 0.28, size * 1.0, material)
    return pole


def add_text(name: str, collection, body: str, x: float, y: float, z: float, size: float, material, align="LEFT"):
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = body
    curve.size = size
    curve.align_x = align
    curve.extrude = size * 0.02
    obj = bpy.data.objects.new(name, curve)
    obj.location = (x, y, z)
    obj.rotation_euler = (1.5708, 0.0, 0.0)  # stand upright, facing +Y camera side
    collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj


# --------------------------------------------------------------------------- #
# scene assembly
# --------------------------------------------------------------------------- #


def build_block(
    scene,
    block_name: str,
    buildings,
    roads,
    hospitals,
    population,
    offset_x: float,
    min_x: float,
    min_y: float,
    scale: float,
    visual: dict,
    highlight_building_ids: set,
    highlight_road_ids: set,
    highlight_hospital_ids: set,
):
    collection = bpy.data.collections.new(block_name)
    scene.collection.children.link(collection)

    def tx(x):
        return (x - min_x) * scale + offset_x

    def ty(y):
        return (y - min_y) * scale

    damage_color = visual["damage_color"]
    height_mult = visual["damage_height_multiplier"]
    road_color = visual["road_color"]
    highlight_color = tuple(visual["targeted_highlight_color"])

    for b in buildings:
        state = b["damage_state"]
        color = tuple(damage_color[state])
        height = max(0.5, b["height_m"] * scale * height_mult[state])
        side = max(0.3, b["side_m"] * scale)
        mat = get_material(f"building_{state}", color, emission=0.45)
        add_box(f"{block_name}_{b['id']}", collection, tx(b["x"]), ty(b["y"]), 0.0, side, side, height, mat)
        if b["id"] in highlight_building_ids:
            hl_mat = get_material("highlight_building", highlight_color, emission=1.5)
            add_box(
                f"{block_name}_{b['id']}_hl", collection,
                tx(b["x"]), ty(b["y"]), height + 0.2, side * 1.15, side * 1.15, 0.4, hl_mat,
            )

    for r in roads:
        state = r["link_state"]
        color = tuple(road_color[state])
        width = {"ARTERIAL": 1.6, "COLLECTOR": 1.1, "LOCAL": 0.7}.get(r["road_class"], 0.8) * max(scale * 40, 0.5)
        mat = get_material(f"road_{state}", color, emission=0.4)
        add_segment_box(
            f"{block_name}_{r['id']}", collection,
            tx(r["x1"]), ty(r["y1"]), tx(r["x2"]), ty(r["y2"]), width, 0.15, mat,
        )
        if r["id"] in highlight_road_ids:
            hl_mat = get_material("highlight_road", highlight_color, emission=1.5)
            add_segment_box(
                f"{block_name}_{r['id']}_hl", collection,
                tx(r["x1"]), ty(r["y1"]), tx(r["x2"]), ty(r["y2"]), width * 1.6, 0.35, hl_mat,
            )

    hospital_mat = get_material("hospital", tuple(visual["hospital_color"]), emission=1.0)
    hospital_size = max(scale * 120, 2.8)
    for h in hospitals:
        add_hospital_marker(f"{block_name}_{h['id']}", collection, tx(h["x"]), ty(h["y"]), 0.0, hospital_size, hospital_mat)
        if h["id"] in highlight_hospital_ids:
            hl_mat = get_material("highlight_hospital", highlight_color, emission=1.5)
            add_sphere(f"{block_name}_{h['id']}_hl", collection, tx(h["x"]), ty(h["y"]), hospital_size * 1.9, hospital_size * 0.35, hl_mat)

    reachable_mat = get_material("population_reachable", tuple(visual["population_reachable_color"]), emission=0.3)
    unreachable_mat = get_material("population_unreachable", tuple(visual["population_unreachable_color"]), emission=0.9)
    for p in population:
        radius = min(max(scale * (p["count"] ** 0.5) * 0.35, 0.15), 0.9)
        z = radius + 1.2  # float above the buildings so markers are not buried
        mat = unreachable_mat if p["unreachable"] else reachable_mat
        add_sphere(f"{block_name}_{p['id']}", collection, tx(p["x"]), ty(p["y"]), z, radius, mat)

    return collection


def build_legend(scene, collection, x: float, y: float, visual: dict):
    white = get_material("legend_text", (0.95, 0.95, 0.95), emission=1.0)
    lines = [
        ("DAMAGE", None),
        ("  NONE", visual["damage_color"]["NONE"]),
        ("  SLIGHT", visual["damage_color"]["SLIGHT"]),
        ("  MODERATE", visual["damage_color"]["MODERATE"]),
        ("  SEVERE", visual["damage_color"]["SEVERE"]),
        ("  COLLAPSE", visual["damage_color"]["COLLAPSE"]),
        ("ROADS", None),
        ("  OPEN", visual["road_color"]["OPEN"]),
        ("  DEGRADED", visual["road_color"]["DEGRADED"]),
        ("  CLOSED", visual["road_color"]["CLOSED"]),
        ("HOSPITAL", visual["hospital_color"]),
        ("POPULATION (red = unreachable)", visual["population_unreachable_color"]),
        ("INTERVENTION target", visual["targeted_highlight_color"]),
    ]
    step = 4.5
    for i, (label, color) in enumerate(lines):
        z = -i * step
        if color is not None:
            add_box(f"legend_swatch_{i}", collection, x, y, z, 2.4, 2.4, 2.4, get_material(f"legend_{i}", tuple(color), emission=0.6))
            add_text(f"legend_text_{i}", collection, label, x + 3.4, y, z, 2.2, white)
        else:
            add_text(f"legend_text_{i}", collection, label, x, y, z, 2.8, white)


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    scene_json_path, out_blend = parse_args()
    data = json.loads(scene_json_path.read_text(encoding="utf-8"))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    visual = data["visual"]
    scale = visual["scale"]
    gap = visual["block_gap"]
    extent = data["extent"]
    min_x, min_y = extent["min_x"], extent["min_y"]
    width = (extent["max_x"] - min_x) * scale
    height = (extent["max_y"] - min_y) * scale

    intervention = data["intervention"]
    highlight_buildings = set(intervention["targeted_building_ids"])
    highlight_roads = set(intervention["targeted_road_ids"])
    highlight_hospitals = set(intervention["targeted_hospital_ids"])

    blocks = [
        ("MRE_Before", data["buildings_before"], data["roads_before"], set(), set(), set()),
        ("MRE_After", data["buildings_after"], data["roads_after"], set(), set(), set()),
        ("MRE_Intervention", data["buildings_after"], data["roads_after"], highlight_buildings, highlight_roads, highlight_hospitals),
    ]

    title_mat = get_material("title_text", (0.95, 0.95, 0.95), emission=1.2)
    for i, (name, buildings, roads, hl_b, hl_r, hl_h) in enumerate(blocks):
        offset_x = i * (width + gap)
        build_block(
            scene, name, buildings, roads, data["hospitals"], data["population"],
            offset_x, min_x, min_y, scale, visual, hl_b, hl_r, hl_h,
        )
        title_collection = bpy.data.collections.get(name)
        add_text(f"{name}_title", title_collection, BLOCK_TITLES[i], offset_x, -10.0, 0.0, 6.0, title_mat)

    legend_collection = bpy.data.collections.new("MRE_Legend")
    scene.collection.children.link(legend_collection)
    total_width = 3 * width + 2 * gap
    build_legend(scene, legend_collection, -gap * 1.6, height * 0.9, visual)

    best = intervention
    if best["has_best_portfolio"]:
        summary_text = (
            f"Best portfolio: {best['best_portfolio_id']}  "
            f"({'improves' if best['improves_on_baseline'] else 'no improvement'})\n"
            f"Mean benefit: {best['primary_benefit_mean']:+.1f} fewer unreachable "
            f"(P={best['probability_of_improvement']:.0%})\n"
            f"Unreachable population: {best['primary_objective_baseline']:.1f} -> "
            f"{best['primary_objective_after']:.1f}"
        )
    else:
        summary_text = "No feasible portfolio improved on the baseline."
    intervention_collection = bpy.data.collections.get("MRE_Intervention")
    add_text(
        "intervention_summary", intervention_collection, summary_text,
        2 * (width + gap), -18.0, 0.0, 2.6, title_mat,
    )

    disclaimer_collection = bpy.data.collections.new("MRE_Disclaimer")
    scene.collection.children.link(disclaimer_collection)
    add_text(
        "disclaimer", disclaimer_collection, data["meta"]["disclaimer"],
        -gap * 1.6, height + 10.0, 0.0, 2.0, title_mat,
    )

    # Light grey-blue background so black city voids read as night streets,
    # not a broken render, and emissive materials still pop against it.
    world = bpy.data.worlds.new("MRE_World")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.05, 0.06, 0.09, 1.0)
        bg.inputs[1].default_value = 1.0
    scene.world = world

    # Key + fill sun lights so every block is evenly lit regardless of angle.
    key_data = bpy.data.lights.new("MRE_Sun_Key", type="SUN")
    key_data.energy = 4.0
    key_obj = bpy.data.objects.new("MRE_Sun_Key", key_data)
    key_obj.location = (total_width * 0.3, -height * 0.5, height * 2.0)
    key_obj.rotation_euler = (0.9, 0.0, 0.6)
    scene.collection.objects.link(key_obj)

    fill_data = bpy.data.lights.new("MRE_Sun_Fill", type="SUN")
    fill_data.energy = 2.0
    fill_obj = bpy.data.objects.new("MRE_Sun_Fill", fill_data)
    fill_obj.location = (total_width * 0.7, height * 0.8, height * 1.5)
    fill_obj.rotation_euler = (2.3, 0.0, -2.4)
    scene.collection.objects.link(fill_obj)

    # Camera framing the whole layout, including the legend to the left of block 1.
    frame_min_x = -gap * 1.6 - 6.0
    frame_max_x = total_width
    frame_center_x = (frame_min_x + frame_max_x) / 2.0
    frame_span_x = frame_max_x - frame_min_x

    cam_data = bpy.data.cameras.new("MRE_Camera")
    cam_data.lens = 20.0
    cam_obj = bpy.data.objects.new("MRE_Camera", cam_data)
    cam_obj.location = (
        frame_center_x,
        -frame_span_x * 0.62,
        frame_span_x * 0.50,
    )
    scene.collection.objects.link(cam_obj)
    look_at(cam_obj, Vector((frame_center_x, height * 0.35, 0.0)))
    scene.camera = cam_obj

    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"SCENE OK: blender {bpy.app.version_string} saved {out_blend} "
          f"({out_blend.stat().st_size:,} bytes)")

    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        try:
            scene.render.engine = "BLENDER_EEVEE"
        except TypeError:
            scene.render.engine = "CYCLES"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.filepath = str(out_blend.with_suffix(".png"))
    try:
        bpy.ops.render.render(write_still=True)
        print(f"RENDER OK: {scene.render.filepath}")
    except RuntimeError as exc:
        print(f"RENDER SKIPPED: {exc}")


if __name__ == "__main__":
    main()
