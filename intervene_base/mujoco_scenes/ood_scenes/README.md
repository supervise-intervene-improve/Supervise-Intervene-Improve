# MuJoCo OOD Scenes

Generated scenes are grouped by task:

- `t_shape/`
- `boxes_cups/`
- `wire_spoon/`

Commands:

```bash
conda run -n polymetis python generate_ood_scenes.py --num-scenes 30 --seed 0
conda run -n polymetis python generate_ood_scenes.py --num-scenes 50 --seed 0
conda run -n polymetis python generate_ood_scenes.py --validate-only mujoco_scenes/ood_scenes/t_shape/t_shape_ood_001.xml
```

The boxes/cups generator interprets the requested cup Z-scale intervals as the
third component of each cup visual and collision mesh `scale` attribute. Cup
X/Y mesh scales are preserved so the top-down cup footprint stays circular and
the red-to-green and blue-to-yellow radial fit relationships are unchanged.

`manifest.json` and `manifest.csv` record every accepted sample.
`validation_report.json` records request counts, rejection counts, and the
assumptions used by the generator.
