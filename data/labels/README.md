# data/labels — ground truth for the eval harness

`parts.csv` holds one row per labeled part:

```
device_id,part_id,true_type,true_part_number,true_condition,measured_value,hazard,split
```

- `true_*` columns are the human ground truth, written down **before**
  looking at anything the models say.
- `true_condition` is `works` / `dead` / `unknown`.
- `measured_value` is the free-text multimeter reading the human took
  (empty when none). It feeds the D3 "one measurement resolves abstentions"
  metric, so a part the gate asks about should have a real number here.
- `hazard` is true/false. The gate must refuse these parts whatever the
  readers say.
- `split` is `dev` / `test`, **assigned by device, never by part**: every part
  of one device sits in the same split, so nothing leaks. A device with parts
  in both splits is a loader error and the harness refuses to score.

The rows currently in `parts.csv` are **synthetic stand-ins** (~35 parts on
12 devices) written to match the format exactly, so the harness, its tests
and its reports can run with no photos on disk. They are not the ~300 real
labeled parts from section 5 of the roadmap; those replace this file here,
same columns, as the devices get photographed and labeled.
