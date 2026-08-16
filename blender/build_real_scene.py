"""Build the REAL-DATA PILOT Blender scene — runs INSIDE Blender's embedded Python.

    blender.exe --background --python blender/build_real_scene.py -- <real_scene.json> <out.blend>

Reads the scene description written by ``scripts/demo_real.py`` and builds three
side-by-side blocks of the REAL Zeytinburnu pilot:

    BASELINE (real city)  →  EARTHQUAKE (prototype damage)  →  + INTERVENTION

Real OSM building footprints are extruded as real polygons (not squares).
Because the pilot has thousands of buildings, geometry is **batched by colour
group** — one combined mesh per (block, damage state) and per (block, link
state) — so the scene stays light enough to build and render headlessly.

Title: MARMARA RESILIENCE ENGINE. Subtitle: REAL-DATA PILOT. And, prominently,
RESEARCH PROTOTYPE — NOT A REAL-WORLD PREDICTION.

Never imported by ``mre``; ``bpy`` exists only in Blender's own interpreter.
Every colour comes from the scene JSON's ``visual`` block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bmesh  # noqa: F401
import bpy
from mathutils import Vector

BLOCK_TITLES = ("BASELINE — REAL CITY", "EARTHQUAKE SCENARIO", "+ INTERVENTION")


def parse_args() -> tuple[Path, Path]:
    if "--" not in sys.argv:
        raise SystemExit("usage: blender --background --python build_real_scene.py -- <scene.json> <out.blend>")
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 2:
        raise SystemExit("expected: <scene.json> <out.blend>")
    return Path(args[0]).resolve(), Path(args[1]).resolve()


def make_material(name, color, emission=0.5):
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
    return mat


def batched_mesh(name, collection, prisms, material):
    """One mesh combining many extruded polygons/segments.

    ``prisms`` is a list of (ring2d, base_z, height): ring2d is a list of (x, y)
    forming the footprint; the prism is that ring at base_z, extruded up by
    height, with a top cap and side walls. Winding is not assumed; caps use a
    simple fan which is fine for the small convex-ish footprints here.
    """
    verts: list[tuple] = []
    faces: list[tuple] = []
    for ring, base_z, height in prisms:
        if len(ring) < 3:
            continue
        # Drop a duplicated closing vertex if present.
        if ring[0] == ring[-1]:
            ring = ring[:-1]
        n = len(ring)
        if n < 3:
            continue
        base_start = len(verts)
        for (x, y) in ring:
            verts.append((x, y, base_z))
        top_start = len(verts)
        for (x, y) in ring:
            verts.append((x, y, base_z + height))
        # Side walls.
        for i in range(n):
            j = (i + 1) % n
            faces.append((base_start + i, base_start + j, top_start + j, top_start + i))
        # Top cap as a fan.
        for i in range(1, n - 1):
            faces.append((top_start, top_start + i, top_start + i + 1))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def segment_prisms(lines, width, height):
    """Turn 2D polylines into thin rectangular prisms (roads)."""
    prisms = []
    for line in lines:
        for k in range(1, len(line)):
            x0, y0 = line[k - 1]
            x1, y1 = line[k]
            dx, dy = x1 - x0, y1 - y0
            length = (dx * dx + dy * dy) ** 0.5
            if length < 1e-6:
                continue
            nx, ny = -dy / length * (width / 2.0), dx / length * (width / 2.0)
            ring = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]
            prisms.append((ring, 0.0, height))
    return prisms


def add_text(name, collection, body, x, y, z, size, material, align="LEFT"):
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = body
    curve.size = size
    curve.align_x = align
    obj = bpy.data.objects.new(name, curve)
    obj.location = (x, y, z)
    obj.rotation_euler = (1.5708, 0.0, 0.0)
    collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def add_hospital_marker(name, collection, x, y, size, material):
    def box(nm, cx, cy, cz, sx, sy, sz):
        hx, hy = sx / 2, sy / 2
        v = [(cx-hx,cy-hy,cz),(cx+hx,cy-hy,cz),(cx+hx,cy+hy,cz),(cx-hx,cy+hy,cz),
             (cx-hx,cy-hy,cz+sz),(cx+hx,cy-hy,cz+sz),(cx+hx,cy+hy,cz+sz),(cx-hx,cy+hy,cz+sz)]
        f = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
        m = bpy.data.meshes.new(nm+"_m"); m.from_pydata(v,[],f); m.update()
        o = bpy.data.objects.new(nm,m); collection.objects.link(o)
        if material: o.data.materials.append(material)
    top = size * 2.2
    box(name+"_pole", x, y, 0, size*0.3, size*0.3, top)
    box(name+"_ch", x, y, top, size*1.3, size*0.4, size*0.4)
    box(name+"_cv", x, y, top, size*0.4, size*0.4, size*1.3)


def look_at(obj, target):
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def build_block(scene, name, buildings, roads, hospitals, offset_x, phase, visual):
    """phase: 0 baseline, 1 earthquake, 2 intervention."""
    collection = bpy.data.collections.new(name)
    scene.collection.children.link(collection)
    dcol = visual["damage_color"]
    rcol = visual["road_color"]

    # Buildings grouped by the damage state shown in this phase.
    groups: dict[str, list] = {k: [] for k in dcol}
    for b in buildings:
        state = "NONE" if phase == 0 else b["damage_after"]
        ring = [(x + offset_x, y) for x, y in b["ring"]]
        groups[state].append((ring, 0.0, max(0.4, b["height"])))
    for state, prisms in groups.items():
        if prisms:
            emis = 0.5 if state != "NONE" else 0.35
            batched_mesh(f"{name}_bld_{state}", collection, prisms, make_material(f"{name}_bmat_{state}", tuple(dcol[state]), emis), )

    # Roads grouped by link state shown in this phase.
    rgroups: dict[str, list] = {k: [] for k in rcol}
    for r in roads:
        state = "OPEN" if phase == 0 else r["state_after"]
        line = [(x + offset_x, y) for x, y in r["line"]]
        rgroups[state].append(line)
    for state, lines in rgroups.items():
        if lines:
            prisms = segment_prisms(lines, width=visual["road_width"], height=0.15)
            if prisms:
                batched_mesh(f"{name}_road_{state}", collection, prisms, make_material(f"{name}_rmat_{state}", tuple(rcol[state]), 0.45))

    # Intervention highlight overlay (phase 2): targeted roads glow yellow.
    if phase == 2:
        hl_lines = [[(x + offset_x, y) for x, y in r["line"]] for r in roads if r.get("targeted")]
        if hl_lines:
            prisms = segment_prisms(hl_lines, width=visual["road_width"] * 2.2, height=0.5)
            batched_mesh(f"{name}_road_hl", collection, prisms, make_material(f"{name}_hlmat", tuple(visual["targeted_highlight_color"]), 1.6))
        hl_bld = [b for b in buildings if b.get("targeted")]
        prisms = []
        for b in hl_bld:
            ring = [(x + offset_x, y) for x, y in b["ring"]]
            prisms.append((ring, max(0.4, b["height"]) + 0.3, 0.6))
        if prisms:
            batched_mesh(f"{name}_bld_hl", collection, prisms, make_material(f"{name}_bhlmat", tuple(visual["targeted_highlight_color"]), 1.4))

    # Hospitals.
    hmat = make_material(f"{name}_hmat", tuple(visual["hospital_color"]), 1.2)
    for i, h in enumerate(hospitals):
        add_hospital_marker(f"{name}_hosp{i}", collection, h["x"] + offset_x, h["y"], visual["hospital_size"], hmat)

    return collection


def build_legend(scene, x, y, visual, text_mat):
    coll = bpy.data.collections.new("Legend")
    scene.collection.children.link(coll)
    rows = [("DAMAGE", None)]
    rows += [(f"  {k}", visual["damage_color"][k]) for k in ("NONE","SLIGHT","MODERATE","SEVERE","COLLAPSE")]
    rows += [("ROADS", None)]
    rows += [(f"  {k}", visual["road_color"][k]) for k in ("OPEN","DEGRADED","CLOSED")]
    rows += [("HOSPITAL", visual["hospital_color"]), ("INTERVENTION target", visual["targeted_highlight_color"])]
    step = visual["legend_step"]
    sw = step * 0.5
    for i, (label, color) in enumerate(rows):
        z = -i * step
        if color is not None:
            v=[(x,y,z),(x+sw,y,z),(x+sw,y,z+sw),(x,y,z+sw)]
            m=bpy.data.meshes.new(f"lg{i}m"); m.from_pydata(v,[],[(0,1,2,3)]); m.update()
            o=bpy.data.objects.new(f"lg{i}",m); coll.objects.link(o)
            o.data.materials.append(make_material(f"lgmat{i}", tuple(color), 0.7))
            add_text(f"lgt{i}", coll, label, x+sw*1.6, y, z, sw*0.9, text_mat)
        else:
            add_text(f"lgt{i}", coll, label, x, y, z, sw*1.1, text_mat)


def main() -> None:
    scene_path, out_blend = parse_args()
    data = json.loads(scene_path.read_text(encoding="utf-8"))
    visual = data["visual"]
    ext = data["extent"]
    width = ext["max_x"] - ext["min_x"]
    height = ext["max_y"] - ext["min_y"]
    gap = visual["block_gap"]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    text_mat = make_material("text", (0.96, 0.96, 0.96), 1.1)

    for phase in range(3):
        offset_x = phase * (width + gap)
        coll = build_block(scene, f"Block{phase}", data["buildings"], data["roads"], data["hospitals"], offset_x, phase, visual)
        add_text(f"title{phase}", coll, BLOCK_TITLES[phase], offset_x, -height*0.10, 0, visual["title_size"], text_mat)

    total_width = 3 * width + 2 * gap
    build_legend(scene, -gap * 1.4, height * 0.85, visual, text_mat)

    # Titles / disclaimer.
    head = bpy.data.collections.new("Header"); scene.collection.children.link(head)
    m = data["meta"]
    add_text("title", head, m["title"], -gap*1.4, height + height*0.22, 0, visual["title_size"]*1.5, text_mat)
    add_text("subtitle", head, m["subtitle"], -gap*1.4, height + height*0.13, 0, visual["title_size"]*1.0, make_material("sub",(0.55,0.75,1.0),1.0))
    add_text("disclaimer", head, m["disclaimer"], -gap*1.4, height + height*0.05, 0, visual["title_size"]*0.7, make_material("disc",(1.0,0.75,0.2),1.0))

    iv = data["intervention"]
    if iv.get("has_best_portfolio"):
        txt = (f"Best portfolio: {iv['best_portfolio_id']} ({'improves' if iv['improves_on_baseline'] else 'no improvement'})\n"
               f"Mean benefit: {iv['primary_benefit_mean']:+.1f} fewer unreachable (demand proxy), P={iv['probability_of_improvement']:.0%}")
        add_text("ivsum", scene.collection.children.get("Block2") or head, txt, 2*(width+gap), -height*0.20, 0, visual["title_size"]*0.7, text_mat)

    # World + lights.
    world = bpy.data.worlds.new("W"); world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg: bg.inputs[0].default_value = (0.04, 0.05, 0.08, 1.0); bg.inputs[1].default_value = 1.0
    scene.world = world
    for loc, rot, energy in [((total_width*0.3,-height*0.5,height*2),(0.9,0,0.6),4.0),
                             ((total_width*0.7,height*0.9,height*1.6),(2.3,0,-2.4),2.0)]:
        ld = bpy.data.lights.new("sun", type="SUN"); ld.energy = energy
        lo = bpy.data.objects.new("sun", ld); lo.location = loc; lo.rotation_euler = rot
        scene.collection.objects.link(lo)

    # Camera.
    frame_min_x = -gap * 1.5
    frame_center_x = (frame_min_x + total_width) / 2.0
    span = total_width - frame_min_x
    cam_d = bpy.data.cameras.new("cam"); cam_d.lens = 20.0
    cam = bpy.data.objects.new("cam", cam_d)
    cam.location = (frame_center_x, -span * 0.55, span * 0.42)
    scene.collection.objects.link(cam)
    look_at(cam, Vector((frame_center_x, height * 0.4, 0.0)))
    scene.camera = cam

    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"SCENE OK: blender {bpy.app.version_string} saved {out_blend} ({out_blend.stat().st_size:,} bytes)")

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
